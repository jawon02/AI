import time
import requests
from ultralytics import YOLO
import cv2  # OpenCV 라이브러리 사용

SERVER_URL = "http://127.0.0.1:8000/api/control/"

print("=== AI 모델(.pt) 로딩 중... ===")
model = YOLO("best (3).pt")
print("=== AI 모델 로딩 완료! ===")

def run_ai_simulator():
    print("=== 1초에 1프레임씩 분석하는 실시간 관제 시뮬레이터 시작 ===")
    
    video_path = "53125-472583428.mp4"
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수 확인 (보통 30)
    if fps == 0: 
        fps = 30
        
    frame_interval = int(fps) # 1초마다 한 번씩 프레임 추출하기 위한 간격
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            #print("=== 동영상 재생 완료. 처음부터 다시 반복합니다. ===")
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 영상이 끝나면 무한 반복
            #continue

            print("=== 12초 동영상 분석이 완료되었습니다. 시뮬레이션을 종료합니다. ===")
            break  # <-- 반복을 멈추고 빠져나감

        frame_idx += 1

        # [핵심] 정확히 1초에 해당하는 프레임만 골라서 분석 수행
        if frame_idx % frame_interval != 0:
            continue

        try:
            print(f"\n--------------------------------------------------")
            print(f"[AI 시뮬레이터] 1초 간격 프레임 스냅샷 분석 중 (진행 상황)...")
            
            # 1초에 딱 1장 추출한 프레임(frame)을 YOLO에 전달
            results = model(frame)
            
            car_count = 5 # 기본값
            for r in results:
                detected_boxes = r.boxes
                if detected_boxes is not None:
                    car_count = len(detected_boxes)

            payload = {
                "pedestrian_count": 0,
                "exit_congested": (car_count > 15),
                "accident_detected": False,
                "emergency_detected": False,
                "north_cars": car_count,
                "south_cars": max(2, car_count // 2)
            }
            
            response = requests.post(SERVER_URL, json=payload)
            
            if response.status_code == 200:
                res_json = response.json()
                print(f"[전송 성공] 서버 응답 결과:")
                print(f"  - 감지된 차량 수(AI): {car_count}대")
                print(f"  - 제어 모드: {res_json.get('mode')}")
                print(f"  - 신호 액션: {res_json.get('signal_action')}")
            else:
                print(f"[전송 실패] 상태 코드: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("[오류] 백엔드 서버와 연결할 수 없습니다.")
        except Exception as e:
            print(f"[오류 발생]: {e}")

        # 시연할 때 눈으로 확인하기 좋게 1초 대기
        time.sleep(1)

    cap.release()

if __name__ == "__main__":
    run_ai_simulator()