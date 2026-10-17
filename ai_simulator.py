import time
import requests
import torch
from ultralytics import YOLO
import cv2
import base64
import threading
import numpy as np

SERVER_URL = "http://127.0.0.1:8000/api/control/"

print("=== AI 모델(.pt) 로딩 중... ===")
model = YOLO("best_accident.pt") 
print("=== AI 모델 로딩 완료! ===")

# 모델 웜업
print("=== AI 모델 웜업(Warm-up) 진행 중... ===")
dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
with torch.no_grad():
    model(dummy_img, imgsz=320, verbose=False)
print("=== AI 모델 준비 완료! ===")

latest_frame = None
cached_detections = []
car_count = 0
is_accident_detected = False
is_ai_running = False
is_sending = False
lock = threading.Lock()

def run_ai_inference():
    """백그라운드 AI 추론 (속도/오탐 최적화 및 사고 개별 색상 식별)"""
    global latest_frame, cached_detections, car_count, is_accident_detected, is_ai_running
    
    try:
        with lock:
            if latest_frame is None:
                return
            frame_to_process = latest_frame.copy()

        with torch.no_grad():
            results = model(frame_to_process, imgsz=320, verbose=False, conf=0.45)
            
        new_detections = []
        accident_found = False
        
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, clss):
                    class_name = model.names[int(cls)].lower()
                    is_acc = ("accident" in class_name) or ("crash" in class_name)
                    
                    if is_acc:
                        accident_found = True
                    
                    # (박스 좌표, 신뢰도, 클래스 ID, 사고 여부) 저장
                    new_detections.append((box, conf, cls, is_acc))
        
        with lock:
            cached_detections = new_detections
            car_count = len(cached_detections)
            is_accident_detected = accident_found
    except Exception as e:
        print(f"[AI 분석 예외]: {e}")
    finally:
        with lock:
            is_ai_running = False

def send_payload_worker(url, frame_to_send, car_cnt, accident_status):
    """독립형 비동기 전송 함수 (데몬 스레드 연동)"""
    global is_sending
    try:
        resized_frame = cv2.resize(frame_to_send, (480, 270))
        _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "pedestrian_count": 0,
            "exit_congested": (car_cnt > 15),
            "accident_detected": accident_status,
            "emergency_detected": False,
            "north_cars": car_cnt,
            "south_cars": max(2, car_cnt // 2),
            "image_frame": jpg_as_text
        }
        requests.post(url, json=payload, timeout=0.3)
    except Exception:
        pass
    finally:
        with lock:
            is_sending = False

def run_ai_simulator():
    global latest_frame, is_ai_running, is_sending, cached_detections, car_count, is_accident_detected

    print("=== 백엔드 서버 연결 상태 확인 중... ===")
    is_server_online = True
    try:
        res = requests.get("http://127.0.0.1:8000/", timeout=1.5)
        if res.status_code != 200:
            is_server_online = False
    except Exception:
        is_server_online = False

    if not is_server_online:
        print("[안내] 백엔드 미연결. 로컬 모드로 화면 재생을 시작합니다.")
    else:
        print("[성공] 백엔드 서버와 정상 연결되었습니다!")

    print("=== 실시간 관제 시뮬레이터 시작 ===")
    
    #video_path = "53125-472583428.mp4"
    video_path = "accident.mp4"
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps): 
        fps = 30.0
        
    server_interval = int(fps)
    delay_per_frame = 1.0 / fps
    frame_idx = 0

    cv2.namedWindow("YOLO Traffic Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Traffic Detection", 800, 450)

    start_sim_time = time.time()
    empty_frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        # 프레임 손상 및 디코딩 오류 안전 리셋
        if not ret or frame is None:
            empty_frame_count += 1
            if empty_frame_count < 5:
                time.sleep(0.01)
                continue
            
            total_elapsed = time.time() - start_sim_time
            print(f"=== 동영상 재생 완료 (소요시간: {total_elapsed:.2f}초). 다시 시작합니다. ===")
            cap.release()
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            empty_frame_count = 0
            start_sim_time = time.time()
            time.sleep(0.05)
            continue

        empty_frame_count = 0
        frame_idx += 1

        # 1. AI 비동기 추론
        with lock:
            latest_frame = frame
            ai_ready = not is_ai_running

        if ai_ready:
            with lock:
                is_ai_running = True
            threading.Thread(target=run_ai_inference, daemon=True).start()

        # 2. 화면 표시 (사고: 빨간색, 일반 차량: 파란색)
        with lock:
            current_detections = list(cached_detections)
            current_car_cnt = car_count
            current_accident_status = is_accident_detected

        for box, conf, cls, is_acc in current_detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            # 사고 라벨일 경우 빨간색 (0, 0, 255), 일반 차량은 파란색 (255, 0, 0)
            box_color = (0, 0, 255) if is_acc else (255, 0, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame, 
                label, 
                (x1, max(y1 - 8, 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )

        # 3. 비동기 실시간 프레임 전송
        if is_server_online:
            with lock:
                send_ready = not is_sending
                if send_ready:
                    is_sending = True

            if send_ready:
                threading.Thread(
                    target=send_payload_worker, 
                    args=(SERVER_URL, frame.copy(), current_car_cnt, current_accident_status),
                    daemon=True
                ).start()

        if frame_idx % server_interval == 1:
            print(f"[백엔드 데이터 갱신] 감지 차량: {current_car_cnt}대 | 사고감지: {current_accident_status}")

        cv2.imshow("YOLO Traffic Detection", frame)
        
        # 'q' 키 누를 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 30 FPS 정속 재생 제어
        elapsed_time = time.time() - start_time
        sleep_time = max(0.001, delay_per_frame - elapsed_time)
        time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ai_simulator()