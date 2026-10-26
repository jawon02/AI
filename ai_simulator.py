"""Optimized YOLO multi-camera traffic simulator.

- Processes three video streams with one shared YOLO model.
- Skips inference when another camera is using the model (no waiting queue).
- Sends only the latest available annotated frame from a separate sender thread.
- Uses lower JPEG dimensions/quality and a reduced transmission rate to reduce backend load.
"""

import base64
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URL = "http://172.20.134.174:8000/api/control/"
SERVER_HEALTH_URL = "http://172.20.134.174:8000/docs"
MODEL_PATH = "best (3).pt"

# Keep the current video-to-camera mapping and backend intersection IDs.
CAMERAS_CONFIG = {
    1: {"name": "매봉터널5 ", "video_path": "test1.mp4", "intersection_id": 1, "direction": "north"},
    2: {"name": "광주역 교차로", "video_path": "test2.mp4", "intersection_id": 1, "direction": "south"},
    3: {"name": "서울", "video_path": "test3.mp4", "intersection_id": 1, "direction": "east"},
}

CONF_THRESHOLD = 0.35
INFERENCE_IMAGE_SIZE = 320
TRACKER_CONFIG = "bytetrack.yaml"
FRAME_SKIP = 5  # Run inference at most once per 5 source frames per camera.

# Network: 4 frames/sec per camera, using low-resolution JPEG.
BACKEND_SEND_INTERVAL_SEC = 0.25
JPEG_WIDTH = 320
JPEG_HEIGHT = 180
JPEG_QUALITY = 40

SMOOTHING_WINDOW = 5
ACCIDENT_VOTES_REQUIRED = 3
CONGESTION_THRESHOLD = 15
FALLBACK_VIDEO_PATH = "광주역교차로.mp4"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
        "MODEL_PATH를 pt 파일의 실제 경로로 수정해 주세요."
    )

for camera_id, config in CAMERAS_CONFIG.items():
    if not Path(config["video_path"]).exists() and not Path(FALLBACK_VIDEO_PATH).exists():
        raise FileNotFoundError(
            f"카메라 {camera_id} 영상 파일을 찾을 수 없습니다: {config['video_path']}\n"
            f"대체 영상도 없습니다: {FALLBACK_VIDEO_PATH}"
        )

print("=== YOLO 모델 로딩 중... ===")
model = YOLO(MODEL_PATH)
# Only one thread is permitted to access the shared YOLO predictor at a time.
# Threads acquire this lock without waiting, so a busy model never blocks video output.
model_lock = threading.Lock()
print("=== YOLO 모델 로딩 완료 ===")
print("클래스 이름:", model.names)

print("=== 모델 웜업 중... ===")
with torch.inference_mode():
    model(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=INFERENCE_IMAGE_SIZE, verbose=False)
print("=== 모델 웜업 완료 ===")

is_running = True


def is_car_class(class_name: str) -> bool:
    return class_name.strip().lower() == "car"


def is_accident_class(class_name: str) -> bool:
    name = class_name.strip().lower()
    return "accident" in name or "crash" in name


def median_count(history, new_value: int) -> int:
    history.append(int(new_value))
    return int(round(float(np.median(list(history)))))


class CameraStreamer(threading.Thread):
    """Read/analyze one video; network sending runs independently."""

    def __init__(self, camera_id, config, server_online):
        super().__init__(daemon=True, name=f"camera-{camera_id}-processor")
        self.camera_id = camera_id
        self.intersection_id = config["intersection_id"]
        self.camera_name = config["name"]
        self.direction = config["direction"]
        self.server_online = server_online

        requested_path = config["video_path"]
        if Path(requested_path).exists():
            self.video_path = requested_path
        else:
            print(
                f"[경고] 카메라 {camera_id} 영상({requested_path})이 없어 "
                f"'{FALLBACK_VIDEO_PATH}'로 대체합니다."
            )
            self.video_path = FALLBACK_VIDEO_PATH

        self.total_count_history = deque(maxlen=SMOOTHING_WINDOW)
        self.accident_history = deque(maxlen=SMOOTHING_WINDOW)
        self.stable_cars = 0
        self.stable_accident = False

        # Single-slot latest-frame mailbox: old unsent frames are overwritten,
        # rather than building up a queue and arriving late at the frontend.
        self.latest_lock = threading.Lock()
        self.latest_frame = None
        self.latest_metrics = None
        self.latest_version = 0
        self.sender_thread = None

    def publish_latest(self, frame, metrics):
        with self.latest_lock:
            self.latest_frame = frame.copy()
            self.latest_metrics = dict(metrics)
            self.latest_version += 1

    def sender_loop(self):
        """Encode/send the newest frame at a limited rate, independent of inference."""
        session = requests.Session()
        last_sent_version = -1
        last_log_time = time.monotonic()
        successful_sends = 0

        try:
            while is_running:
                time.sleep(0.02)
                now = time.monotonic()
                if now - last_log_time < BACKEND_SEND_INTERVAL_SEC:
                    continue

                with self.latest_lock:
                    if self.latest_frame is None or self.latest_metrics is None:
                        continue
                    version = self.latest_version
                    if version == last_sent_version:
                        continue
                    frame = self.latest_frame.copy()
                    metrics = dict(self.latest_metrics)

                # Mark the snapshot consumed before network I/O. A newer frame can
                # replace it while this request is in progress.
                last_sent_version = version
                last_log_time = now

                try:
                    resized = cv2.resize(frame, (JPEG_WIDTH, JPEG_HEIGHT), interpolation=cv2.INTER_AREA)
                    ok, jpeg_buffer = cv2.imencode(
                        ".jpg", resized,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )
                    if not ok:
                        print(f"[카메라 {self.camera_id}] JPEG 인코딩 실패")
                        continue

                    image_frame = base64.b64encode(jpeg_buffer.tobytes()).decode("ascii")
                    count = int(metrics["vehicle_count"])
                    payload = {
                        "intersection_id": self.intersection_id,
                        "camera_id": self.camera_id,
                        "pedestrian_count": 0,
                        "exit_congested": False,
                        "accident_detected": bool(metrics["accident_status"]),
                        "emergency_detected": False,
                        # Keep the original backend payload schema/semantics.
                        "north_cars": 0,
                        "south_cars": 0,
                        "east_cars": count,
                        "west_cars": 0,
                        "vehicle_count": count,
                        "image_frame": image_frame,
                    }

                    response = session.post(
                        SERVER_URL,
                        json=payload,
                        timeout=(1.0, 2.0),
                    )
                    response.raise_for_status()
                    successful_sends += 1
                    if successful_sends == 1 or successful_sends % 20 == 0:
                        print(
                            f"[송출 성공] 카메라={self.camera_id} | "
                            f"차량={count} | 사고={payload['accident_detected']} | "
                            f"누적 전송={successful_sends}"
                        )
                except requests.RequestException as exc:
                    print(f"[카메라 {self.camera_id} 백엔드 전송 실패] {exc}")
                except Exception as exc:
                    print(f"[카메라 {self.camera_id} 프레임 인코딩/전송 예외] {exc}")
        finally:
            session.close()

    def run(self):
        global is_running
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[오류] 카메라 {self.camera_id} 동영상을 열 수 없습니다: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 25.0
        frame_delay = 1.0 / fps
        frame_count = 0
        cached_detections = []
        cached_car_count = 0
        cached_accident = False

        print(
            f"[카메라 {self.camera_id}] {self.camera_name} | "
            f"영상={self.video_path} | FPS={fps:.1f}"
        )

        if self.server_online:
            self.sender_thread = threading.Thread(
                target=self.sender_loop,
                daemon=True,
                name=f"camera-{self.camera_id}-sender",
            )
            self.sender_thread.start()

        try:
            while is_running:
                loop_start = time.monotonic()
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.1)
                        continue
                    # Reset temporal smoothing when the video loops.
                    self.total_count_history.clear()
                    self.accident_history.clear()
                    cached_detections = []
                    cached_car_count = 0
                    cached_accident = False
                    self.stable_cars = 0
                    self.stable_accident = False

                frame_count += 1

                # Do not wait for the shared model lock. If another camera is
                # inferencing, this frame is skipped and playback can continue.
                should_infer = (frame_count % FRAME_SKIP == 0)
                got_model_lock = should_infer and model_lock.acquire(blocking=False)
                if got_model_lock:
                    try:
                        with torch.inference_mode():
                            results = model.track(
                                source=frame,
                                imgsz=INFERENCE_IMAGE_SIZE,
                                conf=CONF_THRESHOLD,
                                persist=True,
                                tracker=TRACKER_CONFIG,
                                verbose=False,
                            )

                        frame_detections = []
                        car_boxes = 0
                        accident_in_frame = False
                        for result in results:
                            boxes_obj = result.boxes
                            if boxes_obj is None or len(boxes_obj) == 0:
                                continue
                            boxes = boxes_obj.xyxy.cpu().numpy()
                            confidences = boxes_obj.conf.cpu().numpy()
                            class_ids = boxes_obj.cls.cpu().numpy().astype(int)
                            track_ids = (
                                boxes_obj.id.cpu().numpy().astype(int)
                                if boxes_obj.id is not None
                                else np.full(len(boxes), -1, dtype=int)
                            )

                            for box, confidence, class_id, track_id in zip(
                                boxes, confidences, class_ids, track_ids
                            ):
                                class_name = str(model.names[int(class_id)])
                                accident_label = is_accident_class(class_name)
                                frame_detections.append(
                                    (box.copy(), float(confidence), class_name,
                                     accident_label, int(track_id))
                                )
                                if accident_label:
                                    accident_in_frame = True
                                if is_car_class(class_name):
                                    # Count detected car boxes for this frame. NMS
                                    # already removes most duplicate detections.
                                    car_boxes += 1

                        cached_detections = frame_detections
                        cached_car_count = car_boxes
                        cached_accident = accident_in_frame
                        self.stable_cars = median_count(
                            self.total_count_history, cached_car_count
                        )
                        self.accident_history.append(bool(cached_accident))
                        self.stable_accident = (
                            sum(self.accident_history) >= ACCIDENT_VOTES_REQUIRED
                        )
                    except Exception as exc:
                        print(f"[카메라 {self.camera_id} AI 추론 예외] {exc}")
                    finally:
                        model_lock.release()

                # Draw the most recent inference on the current frame.
                for box, confidence, class_name, is_accident, track_id in cached_detections:
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 0, 255) if is_accident else (255, 0, 0)
                    id_text = f" ID:{track_id}" if track_id >= 0 else ""
                    label = f"{'accident' if is_accident else class_name} {confidence:.2f}{id_text}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        cv2.LINE_AA,
                    )

                status_line = (
                    f"Intersection:{self.intersection_id} | Cam:{self.camera_id} | "
                    f"Cars:{self.stable_cars} | Accident:{self.stable_accident}"
                )
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (25, 25, 25), -1)
                cv2.putText(
                    frame, status_line, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA,
                )

                metrics = {
                    "vehicle_count": self.stable_cars,
                    "accident_status": self.stable_accident,
                }
                if self.server_online:
                    self.publish_latest(frame, metrics)

                # Keep playback close to the source FPS when inference allows it.
                remaining = frame_delay - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            cap.release()
            if self.sender_thread and self.sender_thread.is_alive():
                self.sender_thread.join(timeout=2.5)
            print(f"[카메라 {self.camera_id}] 종료")


def check_server():
    print("=== 백엔드 서버 연결 상태 확인 중... ===")
    try:
        response = requests.get(SERVER_HEALTH_URL, timeout=(1.0, 2.0))
        if response.status_code == 200:
            print("[성공] 백엔드 서버 연결 성공")
            return True
        print(f"[안내] 서버 응답 코드={response.status_code}. 로컬 모드로 실행합니다.")
    except requests.RequestException as exc:
        print(f"[안내] 백엔드 연결 실패: {exc}. 로컬 모드로 실행합니다.")
    return False


def main():
    global is_running
    server_online = check_server()
    streamers = []

    for camera_id, config in CAMERAS_CONFIG.items():
        streamer = CameraStreamer(camera_id, config, server_online)
        streamers.append(streamer)
        streamer.start()

    print("\n=== 3개 카메라 시뮬레이터 실행 중 (종료: Ctrl+C) ===")
    print(
        f"추론: {INFERENCE_IMAGE_SIZE}px, {FRAME_SKIP}프레임마다 시도 | "
        f"백엔드 전송: 카메라당 최대 {1 / BACKEND_SEND_INTERVAL_SEC:.1f}회/초 | "
        f"JPEG={JPEG_WIDTH}x{JPEG_HEIGHT}, quality={JPEG_QUALITY}"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n=== 시뮬레이터 종료 요청 ===")
    finally:
        is_running = False
        for streamer in streamers:
            streamer.join(timeout=3.0)
        print("=== 시뮬레이터 종료 완료 ===")


if __name__ == "__main__":
    main()