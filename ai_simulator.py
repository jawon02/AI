"""YOLOv8 Multi-Camera Traffic Simulator
Simultaneously tracks 3 video streams (intersection_id: 1, 2, 3) and sends
annotated JPEG frames plus metrics to the FastAPI backend.
"""

import time
import threading
import base64
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 사용자 환경 설정 (Configuration)
# ---------------------------------------------------------------------------
SERVER_URL = "https://resident-washable-facility.ngrok-free.dev/api/control/"
SERVER_HEALTH_URL = "https://resident-washable-facility.ngrok-free.dev/docs"

MODEL_PATH = "best (3).pt"

# SUMO 연동을 위해 각 CCTV를 독립된 교차로(intersection_id 1, 2, 3)로 매핑
CAMERAS_CONFIG = {
    1: {"name": "광주역 교차로 (북측)", "video_path": "test1.mp4", "intersection_id": 1, "direction": "north"},
    2: {"name": "광주역 교차로 (남측)", "video_path": "test2.mp4", "intersection_id": 1, "direction": "south"},
    3: {"name": "광주역 교차로 (동측)", "video_path": "test3.mp4", "intersection_id": 1, "direction": "east"},
}

CONF_THRESHOLD = 0.7
TRACKER_CONFIG = "bytetrack.yaml"

# ⚡ 성능 및 딜레이 최적화 설정
BACKEND_SEND_INTERVAL_SEC = 0.08  # 전송 주기 (초당 약 12~15회 송출)
JPEG_WIDTH = 400
JPEG_HEIGHT = 225
JPEG_QUALITY = 50                 # 화질 조절로 네트워크 병목 제거
FRAME_SKIP = 3                    # 3프레임당 1번만 AI 추론 실행 (속도 저하 방지)

SMOOTHING_WINDOW = 5
ACCIDENT_VOTES_REQUIRED = 3
CONGESTION_THRESHOLD = 15

# ---------------------------------------------------------------------------
# 모델 로딩 및 전역 변수
# ---------------------------------------------------------------------------
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
        "MODEL_PATH를 pt 파일의 실제 경로로 수정해 주세요."
    )

print("=== AI 모델(.pt) 로딩 중... ===")
model = YOLO(MODEL_PATH)
model_lock = threading.Lock()  # 멀티스레드 YOLO 추론 안전용 락
print("=== AI 모델 로딩 완료! ===")

print("=== AI 모델 웜업 진행 중... ===")
dry_run = np.zeros((320, 320, 3), dtype=np.uint8)
with torch.inference_mode():
    model(dry_run, imgsz=320, verbose=False)
print("=== AI 모델 준비 완료! ===")

is_running = True


def is_car_class(class_name):
    return class_name.strip().lower() == "car"


def is_accident_class(class_name):
    name = class_name.strip().lower()
    return ("accident" in name) or ("crash" in name)


# ---------------------------------------------------------------------------
# 카메라 단일 스트림 처리 클래스
# ---------------------------------------------------------------------------
class CameraStreamer(threading.Thread):
    def __init__(self, camera_id, config, server_online):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.intersection_id = config["intersection_id"]
        self.camera_name = config["name"]
        self.direction = config["direction"]
        
        # 파일 존재 확인 후 기본 영상 분기
        raw_path = config["video_path"]
        if Path(raw_path).exists():
            self.video_path = raw_path
        else:
            print(f"[경고] 카메라 {camera_id} 영상 파일({raw_path}) 없음 -> '광주역교차로.mp4'로 대체")
            self.video_path = "광주역교차로.mp4"

        self.server_online = server_online
        self.total_count_history = deque(maxlen=SMOOTHING_WINDOW)
        self.accident_history = deque(maxlen=SMOOTHING_WINDOW)
        self.first_loop_done = False

    def run(self):
        global is_running
        session = requests.Session()
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[오류] 카메라 {self.camera_id} 동영상을 열 수 없음: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0
        delay_per_frame = 1.0 / fps

        print(f"[카메라 {self.camera_id}] 교차로={self.intersection_id} | {self.camera_name} ({self.video_path})")

        last_send_time = 0
        frame_count = 0

        # 캐싱된 추론 결과 (FRAME_SKIP 적용용)
        cached_detections = []
        cached_car_ids = set()
        cached_accident = False

        while is_running:
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                self.first_loop_done = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

            frame_count += 1

            # 1. AI 추론 (FRAME_SKIP 마다 실행하여 CPU/GPU 병목 제거)
            if frame_count % FRAME_SKIP == 0:
                frame_detections = []
                all_car_track_ids = set()
                accident_in_frame = False

                try:
                    with model_lock:
                        with torch.inference_mode():
                            results = model.track(
                                source=frame,
                                imgsz=320,  # 속도 최적화
                                conf=CONF_THRESHOLD,
                                persist=True,
                                tracker=TRACKER_CONFIG,
                                verbose=False,
                            )

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
                                (box, float(confidence), class_name, accident_label, int(track_id))
                            )

                            if accident_label:
                                accident_in_frame = True

                            if is_car_class(class_name) and int(track_id) >= 0:
                                all_car_track_ids.add(int(track_id))

                    cached_detections = frame_detections
                    cached_car_ids = all_car_track_ids
                    cached_accident = accident_in_frame

                except Exception as exc:
                    print(f"[카메라 {self.camera_id} AI 추론 예외] {exc}")

            # 2. 수치 스무딩 (Smoothing)
            raw_total = len(cached_car_ids)
            self.total_count_history.append(raw_total)
            stable_cars = int(round(float(np.median(list(self.total_count_history)))))

            self.accident_history.append(cached_accident)
            stable_accident = sum(self.accident_history) >= ACCIDENT_VOTES_REQUIRED

            # 3. 바운딩 박스 시각화
            for box, confidence, class_name, is_accident, track_id in cached_detections:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if is_accident else (255, 0, 0)
                label_text = f"{'accident' if is_accident else class_name} {confidence:.2f} ID:{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label_text, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )

            status_line = f"Intersection:{self.intersection_id} | Cam:{self.camera_id} | Cars: {stable_cars}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (25, 25, 25), -1)
            cv2.putText(
                frame, status_line, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            # 4. 방향별 차량 수 매핑 (SUMO 연동 데이터 전송)
            north = stable_cars if self.direction == "north" else 0
            south = stable_cars if self.direction == "south" else 0
            east = stable_cars if self.direction == "east" else 0
            west = stable_cars if self.direction == "west" else 0

            # 5. 백엔드 네트워크 전송
            now = time.time()
            if self.server_online and (now - last_send_time >= BACKEND_SEND_INTERVAL_SEC):
                last_send_time = now
                try:
                    resized_frame = cv2.resize(frame, (JPEG_WIDTH, JPEG_HEIGHT))
                    encoded_ok, jpeg_buffer = cv2.imencode(
                        ".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )

                    if encoded_ok:
                        image_frame = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

                        payload = {
                            "intersection_id": self.intersection_id,  # [수정] 1, 2, 3으로 송출
                            "camera_id": self.camera_id,
                            "pedestrian_count": 0,
                            "exit_congested": stable_cars > CONGESTION_THRESHOLD,
                            "accident_detected": stable_accident,
                            "emergency_detected": False,
                            "north_cars": 0,
                            "south_cars": 0,
                            "east_cars": int(stable_cars * 0.9),
                            "west_cars": stable_cars - int(stable_cars * 0.9),
                            "vehicle_count": stable_cars,
                            "image_frame": image_frame,
                        }

                        resp = session.post(SERVER_URL, json=payload, timeout=(0.5, 1.0))
                        resp.raise_for_status()
                        print(f"[송출 성공] 교차로={self.intersection_id}, 카메라={self.camera_id} | 차량={stable_cars}")

                except Exception as exc:
                    print(f"[카메라 {self.camera_id} 백엔드 전송 실패] {exc}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.001, delay_per_frame - elapsed))

        cap.release()
        session.close()


def check_server():
    print("=== 백엔드 서버 연결 상태 확인 중... ===")
    try:
        response = requests.get(SERVER_HEALTH_URL, timeout=1.5)
        if response.status_code == 200:
            print("[성공] 백엔드 서버 연결 성공!")
            return True
        print(f"[안내] 서버 응답 코드: {response.status_code}. 로컬 테스트 모드로 동작합니다.")
    except requests.RequestException as exc:
        print(f"[안내] 백엔드 연결 실패: {exc}. 로컬 테스트 모드로 동작합니다.")
    return False


def main():
    global is_running
    server_online = check_server()

    streamers = []
    for cam_id, config in CAMERAS_CONFIG.items():
        streamer = CameraStreamer(cam_id, config, server_online)
        streamers.append(streamer)
        streamer.start()

    print("\n=== 멀티 교차로/카메라 실시간 시뮬레이터 구동 중 (종료: Ctrl+C) ===")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n=== 시뮬레이터 종료 요청을 수신했습니다. ===")
    finally:
        is_running = False
        for s in streamers:
            s.join(timeout=1.0)
        print("=== 시뮬레이터 정상 종료 완료 ===")


if __name__ == "__main__":
    main()