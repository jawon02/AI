"""ROI (관심 영역) 로직 검증 및 시각화용 multi-camera traffic simulator.

- 차량 바운딩 박스가 ROI 다각형 영역과 1픽셀이라도 겹치거나 경계선에 걸치면 ROI 내 차량으로 집계합니다.
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
# 1. 설정 (Configuration)
# ---------------------------------------------------------------------------
SERVER_URL = "http://172.20.134.174:8000/api/control/"
SERVER_HEALTH_URL = "http://172.20.134.174:8000/docs"
MODEL_PATH = "best (3).pt"

CAMERAS_CONFIG = {
    1: {
        "name": "매봉터널5",
        "video_path": "test1.mp4",
        "intersection_id": 1,
        "direction": "north",
        "roi_ratios": np.array(
            [[0.62, 0.35], [0.76, 0.33], [0.50, 0.98], [0.10, 0.98]],
            dtype=np.float32,
        ),
    },
    2: {
        "name": "광주역 교차로",
        "video_path": "test2.mp4",
        "intersection_id": 1,
        "direction": "south",
        "roi_ratios": np.array(
            [[0.49, 0.25], [0.61, 0.26], [0.42, 0.98], [0.04, 0.97]],
            dtype=np.float32,
        ),
    },
    3: {
        "name": "서울",
        "video_path": "test3.mp4",
        "intersection_id": 1,
        "direction": "east",
        "roi_ratios": np.array(
            [[0.49, 0.25], [0.50, 0.22], [0.54, 0.98], [0.02, 0.98]],
            dtype=np.float32,
        ),
    },
}

CONF_THRESHOLD = 0.25
INFERENCE_IMAGE_SIZE = 640
TRACKER_CONFIG = "bytetrack.yaml"
FRAME_SKIP = 5

BACKEND_SEND_INTERVAL_SEC = 0.25
JPEG_WIDTH = 640
JPEG_HEIGHT = 360
JPEG_QUALITY = 75

SMOOTHING_WINDOW = 5
ACCIDENT_VOTES_REQUIRED = 3
FALLBACK_VIDEO_PATH = "광주역교차로.mp4"

# ---------------------------------------------------------------------------
# 2. 모델 로드 및 웜업
# ---------------------------------------------------------------------------
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
        "MODEL_PATH를 pt 파일의 실제 경로로 수정해 주세요."
    )

for camera_id, config in CAMERAS_CONFIG.items():
    if (
        not Path(config["video_path"]).exists()
        and not Path(FALLBACK_VIDEO_PATH).exists()
    ):
        raise FileNotFoundError(
            f"카메라 {camera_id} 영상 파일을 찾을 수 없습니다: {config['video_path']}\n"
            f"대체 영상도 없습니다: {FALLBACK_VIDEO_PATH}"
        )

print("=== YOLO 모델 로딩 중... ===")
model = YOLO(MODEL_PATH)
model_lock = threading.Lock()
print("=== YOLO 모델 로딩 완료 ===")
print("클래스 이름:", model.names)

print("=== 모델 웜업 중... ===")
with torch.inference_mode():
    model(
        np.zeros((640, 640, 3), dtype=np.uint8),
        imgsz=INFERENCE_IMAGE_SIZE,
        verbose=False,
    )
print("=== 모델 웜업 완료 ===")

is_running = True


def is_car_class(class_name: str) -> bool:
    name = class_name.strip().lower()
    return name in ["car", "vehicle", "bus", "truck", "0"]


def is_accident_class(class_name: str) -> bool:
    name = class_name.strip().lower()
    return "accident" in name or "crash" in name


def median_count(history, new_value: int) -> int:
    history.append(int(new_value))
    return int(round(float(np.median(list(history)))))


# ---------------------------------------------------------------------------
# 3. 카메라 스레드 클래스
# ---------------------------------------------------------------------------
class CameraStreamer(threading.Thread):

    def __init__(self, camera_id, config, server_online):
        super().__init__(daemon=True, name=f"camera-{camera_id}-processor")
        self.camera_id = camera_id
        self.intersection_id = config["intersection_id"]
        self.camera_name = config["name"]
        self.direction = config["direction"]
        self.roi_ratios = config["roi_ratios"]
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
        self.stable_roi_cars = 0
        self.stable_accident = False

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
                    if (
                        self.latest_frame is None
                        or self.latest_metrics is None
                    ):
                        continue
                    version = self.latest_version
                    if version == last_sent_version:
                        continue
                    frame = self.latest_frame.copy()
                    metrics = dict(self.latest_metrics)

                last_sent_version = version
                last_log_time = now

                try:
                    resized = cv2.resize(
                        frame,
                        (JPEG_WIDTH, JPEG_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
                    ok, jpeg_buffer = cv2.imencode(
                        ".jpg",
                        resized,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )
                    if not ok:
                        continue

                    image_frame = base64.b64encode(jpeg_buffer.tobytes()).decode(
                        "ascii"
                    )
                    roi_count = int(metrics["roi_vehicle_count"])

                    payload = {
                        "intersection_id": self.intersection_id,
                        "camera_id": self.camera_id,
                        "pedestrian_count": 0,
                        "exit_congested": False,
                        "accident_detected": bool(metrics["accident_status"]),
                        "emergency_detected": False,
                        "north_cars": (
                            roi_count if self.direction == "north" else 0
                        ),
                        "south_cars": (
                            roi_count if self.direction == "south" else 0
                        ),
                        "east_cars": (
                            roi_count if self.direction == "east" else 0
                        ),
                        "west_cars": (
                            roi_count if self.direction == "west" else 0
                        ),
                        "vehicle_count": roi_count,
                        "image_frame": image_frame,
                    }

                    response = session.post(
                        SERVER_URL, json=payload, timeout=(1.0, 2.0)
                    )
                    response.raise_for_status()
                    successful_sends += 1
                    if (
                        successful_sends == 1
                        or successful_sends % 20 == 0
                    ):
                        print(
                            f"[송출 성공] Cam {self.camera_id}({self.direction}) | "
                            f"ROI 대기 차량: {roi_count}대 | 사고: {payload['accident_detected']}"
                        )
                except requests.RequestException:
                    pass
        finally:
            session.close()

    def run(self):
        global is_running
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(
                f"[오류] 카메라 {self.camera_id} 동영상을 열 수 없습니다: {self.video_path}"
            )
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 25.0
        frame_delay = 1.0 / fps
        frame_count = 0
        cached_detections = []
        cached_roi_car_count = 0
        cached_accident = False

        print(
            f"[카메라 {self.camera_id}] {self.camera_name} 준비 완료 (영상: {self.video_path})"
        )

        if self.server_online:
            self.sender_thread = threading.Thread(
                target=self.sender_loop,
                daemon=True,
                name=f"camera-{self.camera_id}-sender",
            )
            self.sender_thread.start()

        window_name = f"Cam {self.camera_id} - {self.camera_name} (Local Preview)"

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
                    self.total_count_history.clear()
                    self.accident_history.clear()
                    cached_detections = []
                    cached_roi_car_count = 0
                    cached_accident = False
                    self.stable_roi_cars = 0
                    self.stable_accident = False

                h, w = frame.shape[:2]
                roi_pts = (
                    self.roi_ratios * np.array([w, h], dtype=np.float32)
                ).astype(np.int32)

                # ROI 영역 마스크 채우기 (ROI 내부/경계선 픽셀 판단용)
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_pts], 255)

                frame_count += 1

                should_infer = frame_count % FRAME_SKIP == 0
                got_model_lock = should_infer and model_lock.acquire(
                    blocking=False
                )

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
                        roi_car_boxes = 0
                        accident_in_frame = False

                        for result in results:
                            boxes_obj = result.boxes
                            if boxes_obj is None or len(boxes_obj) == 0:
                                continue
                            boxes = boxes_obj.xyxy.cpu().numpy()
                            confidences = boxes_obj.conf.cpu().numpy()
                            class_ids = (
                                boxes_obj.cls.cpu().numpy().astype(int)
                            )
                            track_ids = (
                                boxes_obj.id.cpu().numpy().astype(int)
                                if boxes_obj.id is not None
                                else np.full(len(boxes), -1, dtype=int)
                            )

                            for box, confidence, class_id, track_id in zip(
                                boxes, confidences, class_ids, track_ids
                            ):
                                class_name = str(model.names[int(class_id)])
                                accident_label = is_accident_class(
                                    class_name
                                )

                                x1, y1, x2, y2 = map(int, box)
                                x_center = int((x1 + x2) / 2)
                                y_bottom = y2
                                car_base_point = (x_center, y_bottom)

                                # 바운딩 박스가 ROI 구역 내부/경계선과 1픽셀이라도 겹치는지 판단
                                x1_c, y1_c = max(0, x1), max(0, y1)
                                x2_c, y2_c = min(w, x2), min(h, y2)
                                sub_roi = roi_mask[y1_c:y2_c, x1_c:x2_c]
                                is_in_roi = (
                                    np.any(sub_roi > 0)
                                    if sub_roi.size > 0
                                    else False
                                )

                                frame_detections.append(
                                    (
                                        box.copy(),
                                        float(confidence),
                                        class_name,
                                        accident_label,
                                        int(track_id),
                                        car_base_point,
                                        is_in_roi,
                                    )
                                )

                                if accident_label:
                                    accident_in_frame = True

                                if is_car_class(class_name) and is_in_roi:
                                    roi_car_boxes += 1

                        cached_detections = frame_detections
                        cached_roi_car_count = roi_car_boxes
                        cached_accident = accident_in_frame
                        self.stable_roi_cars = median_count(
                            self.total_count_history, cached_roi_car_count
                        )
                        self.accident_history.append(bool(cached_accident))
                        self.stable_accident = (
                            sum(self.accident_history)
                            >= ACCIDENT_VOTES_REQUIRED
                        )
                    except Exception as exc:
                        print(
                            f"[카메라 {self.camera_id} AI 추론 예외] {exc}"
                        )
                    finally:
                        model_lock.release()

                cv2.polylines(
                    frame,
                    [roi_pts],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

                for (
                    box,
                    confidence,
                    class_name,
                    is_accident,
                    track_id,
                    base_pt,
                    in_roi,
                ) in cached_detections:
                    x1, y1, x2, y2 = map(int, box)
                    box_color = (0, 0, 255) if is_accident else (255, 0, 0)
                    id_text = f" ID:{track_id}" if track_id >= 0 else ""
                    label = f"{'accident' if is_accident else class_name} {confidence:.2f}{id_text}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # ROI 내/경계선에 포함된 차량은 빨간 점, 외부는 파란 점 표시
                    pt_color = (0, 0, 255) if in_roi else (255, 0, 0)
                    cv2.circle(frame, base_pt, 5, pt_color, -1)

                status_line = (
                    f"Cam:{self.camera_id}({self.direction}) | "
                    f"ROI Cars:{self.stable_roi_cars} | Accident:{self.stable_accident}"
                )
                cv2.rectangle(
                    frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1
                )
                cv2.putText(
                    frame,
                    status_line,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    is_running = False
                    break

                metrics = {
                    "roi_vehicle_count": self.stable_roi_cars,
                    "accident_status": self.stable_accident,
                }
                if self.server_online:
                    self.publish_latest(frame, metrics)

                remaining = frame_delay - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            cap.release()
            cv2.destroyWindow(window_name)
            if self.sender_thread and self.sender_thread.is_alive():
                self.sender_thread.join(timeout=2.5)
            print(f"[카메라 {self.camera_id}] 종료")


def check_server():
    print("=== 백엔드 서버 연결 상태 확인 중... ===")
    try:
        response = requests.get(SERVER_HEALTH_URL, timeout=(1.0, 2.0))
        if response.status_code == 200:
            print("[성공] 백엔드 서버 연결 확인됨")
            return True
        print(
            f"[안내] 서버 응답 코드={response.status_code}. 로컬 테스트 모드로 실행합니다."
        )
    except requests.RequestException:
        print("[안내] 백엔드 미연결. 로컬 전용 테스트 모드로 실행합니다.")
    return False


def main():
    global is_running
    server_online = check_server()
    streamers = []

    for camera_id, config in CAMERAS_CONFIG.items():
        streamer = CameraStreamer(camera_id, config, server_online)
        streamers.append(streamer)
        streamer.start()

    try:
        while is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n=== 시뮬레이터 종료 요청 ===")
    finally:
        is_running = False
        for streamer in streamers:
            streamer.join(timeout=3.0)
        cv2.destroyAllWindows()
        print("=== 시뮬레이터 종료 완료 ===")


if __name__ == "__main__":
    main()