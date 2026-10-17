import time
import requests
import torch
from ultralytics import YOLO
import cv2
import base64
import threading
import numpy as np
import gc

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

# 글로벌 상태 관리 변수
latest_frame = None
cached_detections = []
car_count = 0
is_accident_detected = False
is_running = True
lock = threading.Lock()

# 네트워크 전송 전용 큐 변수
send_queue_frame = None
send_queue_data = None
send_trigger = False

def check_box_overlap(target_box, all_boxes, overlap_threshold=0.05):
    """
    단독 차량 오탐 방지: 박스가 다른 차량 박스와 실제로 겹쳐 있는지(충돌 상태인지) 검증
    """
    tx1, ty1, tx2, ty2 = target_box
    target_area = (tx2 - tx1) * (ty2 - ty1)
    if target_area <= 0:
        return False

    for other_box in all_boxes:
        if np.array_equal(target_box, other_box):
            continue
        
        ox1, oy1, ox2, oy2 = other_box
        
        # 교집합 영역 계산
        ix1 = max(tx1, ox1)
        iy1 = max(ty1, oy1)
        ix2 = min(tx2, ox2)
        iy2 = min(ty2, oy2)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            # 타겟 박스 면적 대비 5% 이상 겹치면 충돌 가능성 인정
            if (intersection / target_area) > overlap_threshold:
                return True
    return False

# [영구 스레드 1] AI 추론 전용 백그라운드 워커 (충돌 필터링 적용)
def ai_inference_worker():
    global latest_frame, cached_detections, car_count, is_accident_detected, is_running
    
    last_processed_frame = None
    
    while is_running:
        frame_to_process = None
        with lock:
            if latest_frame is not None and latest_frame is not last_processed_frame:
                frame_to_process = latest_frame.copy()
                last_processed_frame = latest_frame

        if frame_to_process is None:
            time.sleep(0.005)
            continue

        try:
            with torch.no_grad():
                results = model(frame_to_process, imgsz=320, verbose=False, conf=0.35)
                
            raw_detections = []
            all_boxes = []
            
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confs, clss):
                        class_name = model.names[int(cls)].lower()
                        is_acc_label = ("accident" in class_name) or ("crash" in class_name)
                        
                        raw_detections.append((box, conf, cls, is_acc_label))
                        all_boxes.append(box)

            filtered_detections = []
            accident_found = False

            # [핵심 보정] 오탐 제거 로직
            for box, conf, cls, is_acc_label in raw_detections:
                if is_acc_label:
                    # 사고 라벨로 예측되었더라도, 다른 차량과 박스가 겹치지 않으면 '단독 통과 차량'으로 판단하여 정상 차량으로 강제 교정
                    is_colliding = check_box_overlap(box, all_boxes)
                    if is_colliding:
                        accident_found = True
                        filtered_detections.append((box, conf, cls, True))
                    else:
                        # 겹침이 없는 혼자 달리는 차는 파란색(일반 차량) 박스로 교정
                        filtered_detections.append((box, conf, cls, False))
                else:
                    filtered_detections.append((box, conf, cls, False))

            with lock:
                cached_detections = filtered_detections
                car_count = len(cached_detections)
                is_accident_detected = accident_found
                
            del results
        except Exception as e:
            print(f"[AI 분석 예외]: {e}")
            time.sleep(0.01)

# [영구 스레드 2] 백엔드 통신 전용 백그라운드 워커
def network_send_worker():
    global send_queue_frame, send_queue_data, send_trigger, is_running
    
    session = requests.Session()
    
    while is_running:
        frame_to_send = None
        data_to_send = None
        
        with lock:
            if send_trigger and send_queue_frame is not None:
                frame_to_send = send_queue_frame.copy()
                data_to_send = dict(send_queue_data)
                send_trigger = False

        if frame_to_send is None:
            time.sleep(0.01)
            continue

        try:
            resized_frame = cv2.resize(frame_to_send, (480, 270))
            _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "pedestrian_count": 0,
                "exit_congested": (data_to_send['car_cnt'] > 15),
                "accident_detected": data_to_send['accident_status'],
                "emergency_detected": False,
                "north_cars": data_to_send['car_cnt'],
                "south_cars": max(2, data_to_send['car_cnt'] // 2),
                "image_frame": jpg_as_text
            }
            session.post(SERVER_URL, json=payload, timeout=0.3)
        except Exception:
            pass

def run_ai_simulator():
    global latest_frame, cached_detections, car_count, is_accident_detected
    global send_queue_frame, send_queue_data, send_trigger, is_running

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

    # 데몬 스레드 가동
    t_ai = threading.Thread(target=ai_inference_worker, daemon=True)
    t_net = threading.Thread(target=network_send_worker, daemon=True)
    t_ai.start()
    t_net.start()

    print("=== 실시간 관제 시뮬레이터 시작 ===")
    
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

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret or frame is None:
                total_elapsed = time.time() - start_sim_time
                print(f"=== 동영상 재생 완료 (소요시간: {total_elapsed:.2f}초). 다시 시작합니다. ===")
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    cap.release()
                    time.sleep(0.05)
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()

                frame_idx = 0
                start_sim_time = time.time()
                gc.collect()
                
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

            frame_idx += 1

            # 1. AI 추론용 최신 프레임 전달
            with lock:
                latest_frame = frame

            # 2. 보정된 바운딩 박스 화면 표시
            with lock:
                current_detections = list(cached_detections)
                current_car_cnt = car_count
                current_accident_status = is_accident_detected

            for box, conf, cls, is_acc in current_detections:
                x1, y1, x2, y2 = map(int, box)
                
                # 라벨 표기 보정 (사고가 아닐 경우 car로 고정)
                display_label = "accident" if is_acc else "car"
                label_text = f"{display_label} {conf:.2f}"
                box_color = (0, 0, 255) if is_acc else (255, 0, 0) # 사고: 빨강, 일반: 파랑
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame, 
                    label_text, 
                    (x1, max(y1 - 8, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )

            # 3. 백엔드 전송 데이터 갱신
            if is_server_online:
                with lock:
                    send_queue_frame = frame
                    send_queue_data = {
                        'car_cnt': current_car_cnt,
                        'accident_status': current_accident_status
                    }
                    send_trigger = True

            if frame_idx % server_interval == 1:
                print(f"[백엔드 데이터 갱신] 감지 차량: {current_car_cnt}대 | 사고감지: {current_accident_status}")

            cv2.imshow("YOLO Traffic Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("=== 사용자에 의해 시뮬레이터를 종료합니다. ===")
                break

            elapsed_time = time.time() - start_time
            sleep_time = max(0.001, delay_per_frame - elapsed_time)
            time.sleep(sleep_time)

    finally:
        is_running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ai_simulator()