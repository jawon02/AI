"""YOLO26 traffic simulator with tracking, temporal smoothing,
and periodic annotated video-frame plus metrics updates to the backend.

IMPORTANT:
- Edit MODEL_PATH and VIDEO_PATH for your local files.
- No ROI is used; counts are whole-frame vehicle counts.
- Sends one annotated JPEG frame plus stabilized metrics about once per second
  using the existing image_frame JSON field.
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
# User configuration
# ---------------------------------------------------------------------------
SERVER_URL = "https://resident-washable-facility.ngrok-free.dev/api/control/"
SERVER_HEALTH_URL = "https://resident-washable-facility.ngrok-free.dev/docs"

MODEL_PATH = "best (3).pt"
VIDEO_PATH = "광주역교차로.mp4"

CONF_THRESHOLD = 0.7
IMAGE_SIZE = 640
TRACKER_CONFIG = "bytetrack.yaml"

# Send one annotated video frame and the latest stable control metrics per interval.
BACKEND_SEND_INTERVAL_SEC = 0.01
JPEG_WIDTH = 480
JPEG_HEIGHT = 270
JPEG_QUALITY = 60

# Number of recent inference results used to stabilize counts/status.
SMOOTHING_WINDOW = 5
ACCIDENT_VOTES_REQUIRED = 3  # accident  must be seen in >=3 of last 5 inference results
CONGESTION_THRESHOLD = 15

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없어: {MODEL_PATH}\n"
        "MODEL_PATH를 best.pt 파일의 실제 경로로 수정해 줘."
    )
if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(
        f"동영상 파일을 찾을 수 없어: {VIDEO_PATH}\n"
        "VIDEO_PATH를 mp4 파일의 실제 경로로 수정해 줘."
    )

print("=== AI 모델(.pt) 로딩 중... ===")
model = YOLO(MODEL_PATH)
print("=== AI 모델 로딩 완료! ===")
print("클래스 이름:", model.names)

print("=== AI 모델 웜업 진행 중... ===")
dry_run = np.zeros((320, 320, 3), dtype=np.uint8)
with torch.inference_mode():
    model(dry_run, imgsz=320, verbose=False)
print("=== AI 모델 준비 완료! ===")

# ---------------------------------------------------------------------------
# Shared state / smoothing buffers
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
is_running = True

latest_frame = None
latest_frame_seq = 0
cached_detections = []  # (xyxy, confidence, class_id, is_accident_class, track_id)

stable_total_cars = 0
stable_accident_detected = False

# Latest annotated frame and metrics read by the backend worker.
latest_backend_frame = None
latest_metrics = None

# Updated only by the inference worker.
total_count_history = deque(maxlen=SMOOTHING_WINDOW)
accident_history = deque(maxlen=SMOOTHING_WINDOW)


def median_count(history, new_value):
    """Add a count sample and return its recent median."""
    history.append(int(new_value))
    return int(round(float(np.median(list(history)))))


def is_car_class(class_name):
    """Count only the normal-car class as a vehicle, not the accident class."""
    return class_name.strip().lower() == "car"


def is_accident_class(class_name):
    name = class_name.strip().lower()
    return ("accident" in name) or ("crash" in name)


def ai_inference_worker():
    """Track the latest frame, count unique car IDs across the whole frame, and smooth metrics."""
    global latest_frame, latest_frame_seq, cached_detections
    global stable_total_cars
    global stable_accident_detected, is_running

    last_processed_seq = -1

    while is_running:
        frame_to_process = None
        this_seq = -1

        with state_lock:
            if latest_frame is not None and latest_frame_seq != last_processed_seq:
                frame_to_process = latest_frame.copy()
                this_seq = latest_frame_seq
                last_processed_seq = latest_frame_seq

        if frame_to_process is None:
            time.sleep(0.003)
            continue

        try:
            with torch.inference_mode():
                results = model.track(
                    source=frame_to_process,
                    imgsz=IMAGE_SIZE,
                    conf=CONF_THRESHOLD,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    verbose=False,
                )

            frame_detections = []
            all_car_track_ids = set()
            accident_in_frame = False

            for result in results:
                boxes_obj = result.boxes
                if boxes_obj is None or len(boxes_obj) == 0:
                    continue

                boxes = boxes_obj.xyxy.cpu().numpy()
                confidences = boxes_obj.conf.cpu().numpy()
                class_ids = boxes_obj.cls.cpu().numpy().astype(int)
                if boxes_obj.id is not None:
                    track_ids = boxes_obj.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.full(len(boxes), -1, dtype=int)

                for box, confidence, class_id, track_id in zip(
                    boxes, confidences, class_ids, track_ids
                ):
                    class_name = str(model.names[int(class_id)])
                    accident_label = is_accident_class(class_name)
                    frame_detections.append(
                        (box.copy(), float(confidence), int(class_id), accident_label,
                         None if int(track_id) < 0 else int(track_id))
                    )

                    if accident_label:
                        accident_in_frame = True

                    # Count the car class only. Count unique tracker IDs to reduce
                    # duplicate detections in a single frame.
                    if is_car_class(class_name) and int(track_id) >= 0:
                        track_id = int(track_id)
                        all_car_track_ids.add(track_id)

            raw_total = len(all_car_track_ids)
            total_smoothed = median_count(total_count_history, raw_total)

            accident_history.append(bool(accident_in_frame))
            accident_smoothed = (
                sum(accident_history) >= ACCIDENT_VOTES_REQUIRED
            )

            with state_lock:
                cached_detections = frame_detections
                stable_total_cars = total_smoothed
                stable_accident_detected = bool(accident_smoothed)

            del results

        except Exception as exc:
            print(f"[AI 분석 예외] {exc}")
            time.sleep(0.02)


def network_send_worker(server_online):
    """Send the latest annotated JPEG frame and stable metrics about once per second."""
    global latest_backend_frame, latest_metrics, is_running

    if not server_online:
        return

    session = requests.Session()
    while is_running:
        time.sleep(BACKEND_SEND_INTERVAL_SEC)
        if not is_running:
            break

        # Take a consistent snapshot of the newest annotated frame and metrics.
        with state_lock:
            frame = latest_backend_frame.copy() if latest_backend_frame is not None else None
            metrics = dict(latest_metrics) if latest_metrics is not None else None

        if frame is None or metrics is None:
            continue

        try:
            # Send a compressed frame every interval, not every source-video frame.
            resized_frame = cv2.resize(frame, (JPEG_WIDTH, JPEG_HEIGHT))
            encoded_ok, jpeg_buffer = cv2.imencode(
                ".jpg",
                resized_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not encoded_ok:
                print("[백엔드 전송 실패] JPEG 인코딩에 실패했어.")
                continue

            image_frame = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

            cnt = metrics["vehicle_count"]
            payload = {
                "pedestrian_count": 0,
                "exit_congested": cnt > CONGESTION_THRESHOLD,
                "accident_detected": metrics["accident_status"],
                "emergency_detected": False,
                
                # [4방향 차량 수 비중 설정]
                # 남북 방향에 비중을 높게 주어 백엔드에서 CONGESTION_PATTERN이 발동되도록 전달
                "north_cars": cnt,                       # 예: 12대
                "south_cars": max(2, cnt // 2),           # 예: 6대
                "east_cars": max(1, cnt // 3),            # 예: 4대
                "west_cars": max(1, cnt // 4),            # 예: 3대
                
                "vehicle_count": cnt,
                "image_frame": image_frame,
            }



            response = session.post(SERVER_URL, json=payload, timeout=(1.0, 3.0))
            response.raise_for_status()
            print(
                "[백엔드 영상+데이터 전송] "
                f"전체 차량={payload['vehicle_count']} | "
                f"혼잡={payload['exit_congested']} | 사고={payload['accident_detected']} | "
                f"프레임={JPEG_WIDTH}x{JPEG_HEIGHT}"
            )
        except requests.RequestException as exc:
            print(f"[백엔드 전송 실패] {exc}")
        except Exception as exc:
            print(f"[영상 인코딩/전송 예외] {exc}")

    session.close()


def check_server():
    print("=== 백엔드 서버 연결 상태 확인 중... ===")
    try:
        response = requests.get(SERVER_HEALTH_URL, timeout=1.5)
        if response.status_code == 200:
            print("[성공] 백엔드 서버와 연결되었습니다.")
            return True
        print(f"[안내] 서버 응답 코드: {response.status_code}. 로컬 모드로 실행합니다.")
    except requests.RequestException as exc:
        print(f"[안내] 백엔드 연결 실패: {exc}. 로컬 모드로 실행합니다.")
    return False


def run_ai_simulator():
    global latest_frame, latest_frame_seq, latest_backend_frame, latest_metrics, is_running

    is_running = True
    server_online = check_server()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"동영상을 열 수 없어: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    delay_per_frame = 1.0 / fps
    frame_idx = 0

    ai_thread = threading.Thread(target=ai_inference_worker, daemon=True)
    ai_thread.start()

    net_thread = threading.Thread(
        target=network_send_worker, args=(server_online,), daemon=True
    )
    net_thread.start()

    cv2.namedWindow("YOLO Traffic Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Traffic Detection", 1000, 600)

    print("=== 실시간 관제 시뮬레이터 시작 (종료: Q) ===")
    print("ROI 비활성화: 화면 전체에서 추적된 car ID 수를 집계합니다.")
    print(f"백엔드 전송: 매 {BACKEND_SEND_INTERVAL_SEC:.1f}초 JPEG 영상 프레임 + 교통 데이터")
    start_sim_time = time.time()

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                print(
                    f"=== 동영상 재생 완료 ({time.time() - start_sim_time:.2f}초). "
                    "처음부터 다시 재생합니다. ==="
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                start_sim_time = time.time()
                frame_idx = 0

            frame_idx += 1
            with state_lock:
                latest_frame = frame.copy()
                latest_frame_seq += 1
                current_detections = list(cached_detections)
                total_count = stable_total_cars
                accident_status = stable_accident_detected

            # Build latest stable metrics for the independent video+metrics sender.
            metrics_now = {
                "vehicle_count": total_count,
                "accident_status": accident_status,
            }
            # Display the detected bounding boxes locally.
            for box, confidence, class_id, is_accident, track_id in current_detections:
                x1, y1, x2, y2 = map(int, box)
                class_name = str(model.names[int(class_id)]).lower()
                if is_accident:
                    display_label = "accident"
                    color = (0, 0, 255)  # red, BGR
                else:
                    display_label = class_name
                    color = (255, 0, 0)  # blue, BGR

                id_text = f" ID:{track_id}" if track_id is not None else ""
                label_text = f"{display_label} {confidence:.2f}{id_text}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label_text, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

            status_line = (
                f"Cars: {total_count} | Accident: {accident_status}"
            )
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (25, 25, 25), -1)
            cv2.putText(
                frame, status_line, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Publish the annotated frame and its matching metrics atomically.
            with state_lock:
                latest_backend_frame = frame.copy()
                latest_metrics = metrics_now

            if frame_idx % max(1, int(fps)) == 0:
                print(
                    f"[집계] 전체 car={total_count} | 사고={accident_status}"
                )

            cv2.imshow("YOLO Traffic Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("=== 사용자에 의해 시뮬레이터를 종료합니다. ===")
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0.001, delay_per_frame - elapsed))

    finally:
        is_running = False
        cap.release()
        cv2.destroyAllWindows()
        ai_thread.join(timeout=2.0)
        net_thread.join(timeout=2.0)
        print("=== 시뮬레이터 종료 완료 ===")


if __name__ == "__main__":
    run_ai_simulator()