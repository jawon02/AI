import time
import requests
from ultralytics import YOLO
import cv2
import base64
from concurrent.futures import ThreadPoolExecutor

SERVER_URL = "http://127.0.0.1:8000/api/control/"

print("=== AI 모델(.pt) 로딩 중... ===")
model = YOLO("best (3).pt")
print("=== AI 모델 로딩 완료! ===")

# HTTP 세션 재활용 (TCP 커넥션 유지로 속도 향상)
session = requests.Session()
executor = ThreadPoolExecutor(max_workers=2)

def send_payload_async(url, payload):
    try:
        session.post(url, json=payload, timeout=1.0)
    except Exception:
        pass

def run_ai_simulator():
    print("=== 백엔드 서버 연결 상태 확인 중... ===")
    is_server_online = True
    
    try:
        response = session.get("http://127.0.0.1:8000/", timeout=2.0)
        if response.status_code != 200:
            is_server_online = False
    except requests.exceptions.ConnectionError:
        is_server_online = False

    if not is_server_online:
        print("[오류] 백엔드 서버와 연결할 수 없습니다. 3초 후 로컬 모드로 동작합니다.")
        time.sleep(3)
    else:
        print("[성공] 백엔드 서버와 정상적으로 연결되었습니다!")

    print("=== 실시간 관제 시뮬레이터 시작 ===")
    
    video_path = "53125-472583428.mp4"
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: 
        fps = 30.0
        
    server_interval = int(fps)                 # 1초 주기는 제어 로그 기준
    detection_interval = max(1, int(fps / 10))  # 0.1초마다 박스 위치 갱신
    
    delay_per_frame = 1.0 / fps
    frame_idx = 0

    cached_detections = []
    car_count = 5

    # 루프 시작 전 Window 생성 (반복 호출 오버헤드 제거)
    cv2.namedWindow("YOLO Traffic Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Traffic Detection", 800, 450)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("=== 동영상 분석 완료. 재시작합니다. ===")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        frame_idx += 1

        # -------------------------------------------------------------
        # 1. AI 검출 위치 갱신 (0.1초마다 실행, imgsz=640으로 추론 속도 최적화)
        # -------------------------------------------------------------
        if frame_idx % detection_interval == 1 or not cached_detections:
            try:
                results = model(frame, imgsz=640, verbose=False)
                new_detections = []
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        clss = r.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confs, clss):
                            new_detections.append((box, conf, cls))
                
                cached_detections = new_detections
                car_count = len(cached_detections)
            except Exception as e:
                print(f"[AI 분석 오류]: {e}")

        # -------------------------------------------------------------
        # 2. 화면에 바운딩 박스(파란색) 및 텍스트(하얀색) 그리기
        # -------------------------------------------------------------
        for box, conf, cls in cached_detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            # 파란색 바운딩 박스 (BGR: 255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 하얀색 텍스트 (BGR: 255, 255, 255)
            cv2.putText(
                frame, 
                label, 
                (x1, max(y1 - 8, 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255),  # ★ 하얀색 변경 완료
                2
            )

        # -------------------------------------------------------------
        # 3. 백엔드 전송용 리사이즈 및 JPEG 인코딩 (13초 정속 재생 병목 해결)
        # -------------------------------------------------------------
        resized_frame = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "pedestrian_count": 0,
            "exit_congested": (car_count > 15),
            "accident_detected": False,
            "emergency_detected": False,
            "north_cars": car_count,
            "south_cars": max(2, car_count // 2),
            "image_frame": jpg_as_text
        }

        # 비동기 전송
        if is_server_online:
            executor.submit(send_payload_async, SERVER_URL, payload)

        if frame_idx % server_interval == 1:
            print(f"[백엔드 제어 송신 (1초 주기)] 감지 차량: {car_count}대")

        # OpenCV 로컬 창 출력
        cv2.imshow("YOLO Traffic Detection", frame)
        cv2.waitKey(1)

        # 30 FPS 정속 재생 대기 (~0.033초)
        elapsed_time = time.time() - start_time
        sleep_time = max(0.001, delay_per_frame - elapsed_time)
        time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)

if __name__ == "__main__":
    run_ai_simulator()