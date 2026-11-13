"""Multi-Camera Traffic & Incident Simulator (Fixed Detection & Frontend Sync)

- 카메라 1 (혼잡): test1.mp4 + best (3).pt (ROI 사용, 해상도 480 원복)
- 카메라 2 (사고): notest.mp4 + best_accident.pt (사고 전용)
- 카메라 3 (응급차량): emergency.mp4 + best_emergency.pt (응급 전용)
- 카메라 4 (일반): Normal.mp4 + best (3).pt (ROI 사용, 해상도 480 원복)
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

CAMERAS_CONFIG = {
    1: {
        "name": "혼잡",
        "video_path": "test1.mp4",
        "model_paths": ["best (3).pt"],
        "intersection_id": 1,
        "direction": "north",
        "use_roi": True,
        "draw_cars": True,
        "conf_threshold": 0.25,
        "occupancy_boost": 20.0,
        "frame_skip": 3,
        "roi_ratios": np.array(
            [[0.62, 0.35], [0.76, 0.33], [0.50, 0.98], [0.10, 0.98]],
            dtype=np.float32,
        ),
    },
    2: {
        "name": "사고",
        "video_path": "notest.mp4",
        "model_paths": ["best_accident.pt"],
        "intersection_id": 1,
        "direction": "south",
        "use_roi": False,
        "draw_cars": True,
        "conf_threshold": 0.45,
        "occupancy_boost": 0.0,
        "frame_skip": 2,
        "roi_ratios": None,
    },
    3: {
        "name": "응급차량",
        "video_path": "emergency.mp4",
        "model_paths": ["best_emergency.pt"],
        "intersection_id": 1,
        "direction": "east",
        "use_roi": False,
        "draw_cars": False,
        "conf_threshold": 0.40,
        "occupancy_boost": 0.0,
        "frame_skip": 3,
        "roi_ratios": None,
    },
    4: {
        "name": "일반",
        "video_path": "Normal.mp4",
        "model_paths": ["best (3).pt"],
        "intersection_id": 1,
        "direction": "west",
        "use_roi": True,
        "draw_cars": True,
        "conf_threshold": 0.25,
        "occupancy_boost": 0.0,
        "frame_skip": 3,
        "roi_ratios": np.array(
            [[0.30, 0.26], [0.35, 0.25], [0.49, 0.67], [0.32, 0.80]],
            dtype=np.float32,
        ),
    },
}

# 💡 차량 탐지 정확도 향상을 위해 해상도를 480으로 원복
INFERENCE_IMAGE_SIZE = 480

BACKEND_SEND_INTERVAL_SEC = 0.066
JPEG_WIDTH = 480
JPEG_HEIGHT = 270
JPEG_QUALITY = 65

SMOOTHING_WINDOW = 5
ACCIDENT_VOTES_REQUIRED = 2
EMERGENCY_VOTES_REQUIRED = 2
FALLBACK_VIDEO_PATH = "광주역교차로.mp4"

# ---------------------------------------------------------------------------
# 2. 다중 모델 로드 및 웜업
# ---------------------------------------------------------------------------
MODELS = {}
MODEL_LOCKS = {}

required_models = set()
for config in CAMERAS_CONFIG.values():
    for m_path in config["model_paths"]:
        required_models.add(m_path)

for m_path in required_models:
    if not Path(m_path).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {m_path}")
    print(f"=== YOLO 모델 로딩 중: {m_path} ===")
    loaded_model = YOLO(m_path)
    MODELS[m_path] = loaded_model
    MODEL_LOCKS[m_path] = threading.Lock()
    print(f"=== YOLO 모델 로딩 완료: {m_path} ===")

print("=== 전체 모델 웜업 중... ===")
with torch.inference_mode():
    for m_path, m_inst in MODELS.items():
        m_inst(
            np.zeros((480, 480, 3), dtype=np.uint8),
            imgsz=INFERENCE_IMAGE_SIZE,
            verbose=False,
        )
print("=== 전체 모델 웜업 완료 ===")

is_running = True


def is_car_class(class_name: str) -> bool:
    name = class_name.strip().lower()
    return name in ["car", "vehicle", "bus", "truck"]


def is_accident_class(class_name: str) -> bool:
    name = class_name.strip().lower()
    return "accident" in name or "crash" in name or "사고" in name


def is_emergency_class(class_name: str) -> bool:
    name = class_name.strip().lower()
    return any(k in name for k in ["emergency", "ambulance", "fire", "police", "응급", "구급차"])


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
        self.model_paths = config["model_paths"]
        self.use_roi = config["use_roi"]
        self.draw_cars = config["draw_cars"]
        self.conf_threshold = config["conf_threshold"]
        self.frame_skip = config.get("frame_skip", 2)
        self.roi_ratios = config["roi_ratios"]
        self.occupancy_boost = config.get("occupancy_boost", 0.0)
        self.server_online = server_online

        requested_path = config["video_path"]
        if Path(requested_path).exists():
            self.video_path = requested_path
        else:
            self.video_path = FALLBACK_VIDEO_PATH

        self.total_count_history = deque(maxlen=SMOOTHING_WINDOW)
        self.occupancy_history = deque(maxlen=SMOOTHING_WINDOW)
        self.accident_history = deque(maxlen=SMOOTHING_WINDOW)
        self.emergency_history = deque(maxlen=SMOOTHING_WINDOW)

        self.stable_roi_cars = 0
        self.stable_occupancy_rate = 0.0
        self.stable_accident = False
        self.stable_emergency = False

        self.latest_lock = threading.Lock()
        self.latest_frame = None
        self.latest_metrics = None
        self.latest_version = 0
        self.sender_thread = None

        self.infer_lock = threading.Lock()
        self.cached_detections = []
        self.is_inferring = False

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
                time.sleep(0.01)
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

                    image_frame = base64.b64encode(jpeg_buffer.tobytes()).decode("ascii")
                    roi_count = int(metrics["roi_vehicle_count"])
                    occupancy_pct = float(metrics["occupancy_rate"])

                    roi_occupancy_ratio = round(occupancy_pct / 100.0, 4)
                    roi_congestion_threshold = 0.80
                    exit_congested = bool(roi_occupancy_ratio >= roi_congestion_threshold)

                    payload = {
                        "intersection_id": self.intersection_id,
                        "camera_id": self.camera_id,
                        "vehicle_count": roi_count,
                        "roi_occupancy_ratio": roi_occupancy_ratio,
                        "roi_congestion_threshold": roi_congestion_threshold,
                        "exit_congested": exit_congested,
                        "pedestrian_count": 0,
                        "accident_detected": bool(metrics["accident_status"]),
                        "emergency_detected": bool(metrics["emergency_status"]),
                        "north_cars": roi_count if self.direction == "north" else 0,
                        "south_cars": roi_count if self.direction == "south" else 0,
                        "east_cars": roi_count if self.direction == "east" else 0,
                        "west_cars": roi_count if self.direction == "west" else 0,
                        "image_frame": image_frame,
                    }

                    response = session.post(SERVER_URL, json=payload, timeout=(1.0, 2.0))
                    response.raise_for_status()
                    successful_sends += 1
                    if successful_sends == 1 or successful_sends % 50 == 0:
                        print(
                            f"[송출 성공] Cam {self.camera_id}({self.camera_name}) | "
                            f"ROI 차량: {roi_count}대 | 점유율: {roi_occupancy_ratio:.2f} | "
                            f"혼잡: {exit_congested} | 사고: {payload['accident_detected']} | 응급: {payload['emergency_detected']}"
                        )
                except requests.RequestException:
                    pass
        finally:
            session.close()

    def _async_inference(self, frame_crop, h, w, roi_mask, total_roi_pixels):
        try:
            frame_detections = []
            roi_car_boxes = 0
            accident_in_frame = False
            emergency_in_frame = False
            car_mask = np.zeros((h, w), dtype=np.uint8)

            for m_path in self.model_paths:
                model_inst = MODELS[m_path]
                model_lock = MODEL_LOCKS[m_path]

                with model_lock:
                    with torch.inference_mode():
                        results = model_inst.predict(
                            source=frame_crop,
                            imgsz=INFERENCE_IMAGE_SIZE,
                            conf=self.conf_threshold,
                            verbose=False,
                        )

                for result in results:
                    boxes_obj = result.boxes
                    if boxes_obj is None or len(boxes_obj) == 0:
                        continue
                    boxes = boxes_obj.xyxy.cpu().numpy()
                    confidences = boxes_obj.conf.cpu().numpy()
                    class_ids = boxes_obj.cls.cpu().numpy().astype(int)

                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        class_name = str(model_inst.names[int(class_id)])

                        is_accident = is_accident_class(class_name) if "accident" in m_path else False
                        is_emergency = is_emergency_class(class_name) if "emergency" in m_path else False

                        x1, y1, x2, y2 = map(int, box)
                        x_center = int((x1 + x2) / 2)
                        y_bottom = y2
                        car_base_point = (x_center, y_bottom)

                        if self.use_roi and roi_mask is not None:
                            x1_c, y1_c = max(0, x1), max(0, y1)
                            x2_c, y2_c = min(w, x2), min(h, y2)
                            sub_roi = roi_mask[y1_c:y2_c, x1_c:x2_c]
                            is_in_roi = np.any(sub_roi > 0) if sub_roi.size > 0 else False
                        else:
                            is_in_roi = False

                        frame_detections.append(
                            (
                                box.copy(),
                                float(confidence),
                                class_name,
                                is_accident,
                                is_emergency,
                                car_base_point,
                                is_in_roi,
                            )
                        )

                        if is_accident:
                            accident_in_frame = True
                        if is_emergency:
                            emergency_in_frame = True

                        if self.use_roi and is_car_class(class_name) and is_in_roi:
                            roi_car_boxes += 1
                            cv2.rectangle(car_mask, (x1_c, y1_c), (x2_c, y2_c), 255, -1)

            if self.use_roi and total_roi_pixels > 0:
                overlap_mask = cv2.bitwise_and(car_mask, roi_mask)
                car_roi_pixels = cv2.countNonZero(overlap_mask)
                raw_occupancy = (car_roi_pixels / total_roi_pixels) * 100.0
                current_occupancy = min(100.0, raw_occupancy + self.occupancy_boost) if car_roi_pixels > 0 else 0.0
            else:
                current_occupancy = 0.0

            with self.infer_lock:
                self.cached_detections = frame_detections
                self.stable_roi_cars = median_count(self.total_count_history, roi_car_boxes)
                self.occupancy_history.append(float(current_occupancy))
                self.stable_occupancy_rate = round(float(np.median(list(self.occupancy_history))), 1)

                self.accident_history.append(bool(accident_in_frame))
                self.stable_accident = sum(self.accident_history) >= ACCIDENT_VOTES_REQUIRED

                self.emergency_history.append(bool(emergency_in_frame))
                self.stable_emergency = sum(self.emergency_history) >= EMERGENCY_VOTES_REQUIRED

        finally:
            self.is_inferring = False

    def run(self):
        global is_running
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 25.0
        frame_delay = 1.0 / fps
        frame_count = 0

        self.cached_detections = []
        self.stable_roi_cars = 0
        self.stable_occupancy_rate = 0.0
        self.stable_accident = False
        self.stable_emergency = False

        if self.server_online:
            self.sender_thread = threading.Thread(
                target=self.sender_loop, daemon=True, name=f"camera-{self.camera_id}-sender"
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

                h, w = frame.shape[:2]

                if self.use_roi and self.roi_ratios is not None:
                    roi_pts = (self.roi_ratios * np.array([w, h], dtype=np.float32)).astype(np.int32)
                    roi_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(roi_mask, [roi_pts], 255)
                    total_roi_pixels = cv2.countNonZero(roi_mask)
                else:
                    roi_pts, roi_mask, total_roi_pixels = None, None, 0

                frame_count += 1

                if frame_count % self.frame_skip == 0 and not self.is_inferring:
                    self.is_inferring = True
                    threading.Thread(
                        target=self._async_inference,
                        args=(frame.copy(), h, w, roi_mask, total_roi_pixels),
                        daemon=True,
                    ).start()

                if self.use_roi and roi_pts is not None:
                    cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 255, 0), thickness=2)

                with self.infer_lock:
                    detections_to_draw = list(self.cached_detections)

                for (
                    box,
                    confidence,
                    class_name,
                    is_accident,
                    is_emergency,
                    base_pt,
                    in_roi,
                ) in detections_to_draw:
                    if not self.draw_cars and not is_accident and not is_emergency:
                        continue

                    if self.use_roi and not in_roi and not is_accident and not is_emergency:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    box_color = (0, 0, 255) if is_accident else ((0, 255, 255) if is_emergency else (255, 0, 0))
                    label = f"{class_name} {confidence:.2f}"

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

                ratio_val = round(self.stable_occupancy_rate / 100.0, 2)
                status_line = (
                    f"Cam:{self.camera_id}({self.camera_name}) | "
                    f"ROI Cars:{self.stable_roi_cars} ({ratio_val:.2f} / {self.stable_occupancy_rate:.1f}%) | "
                    f"Accident:{self.stable_accident} | Emergency:{self.stable_emergency}"
                )
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
                cv2.putText(
                    frame, status_line, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )

                metrics = {
                    "roi_vehicle_count": self.stable_roi_cars,
                    "occupancy_rate": self.stable_occupancy_rate,
                    "accident_status": self.stable_accident,
                    "emergency_status": self.stable_emergency,
                }
                if self.server_online:
                    self.publish_latest(frame, metrics)

                remaining = frame_delay - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            cap.release()


def check_server():
    try:
        response = requests.get(SERVER_HEALTH_URL, timeout=(1.0, 2.0))
        return response.status_code == 200
    except requests.RequestException:
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
        pass
    finally:
        is_running = False
        for streamer in streamers:
            streamer.join(timeout=3.0)


if __name__ == "__main__":
    main()