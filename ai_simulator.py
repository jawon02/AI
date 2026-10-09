import time
import requests
from ultralytics import YOLO
import cv2  # OpenCV 라이브러리 사용

SERVER_URL = "http://127.0.0.1:8000/api/control/"

print("=== AI 모델(.pt) 로딩 중... ===")
model = YOLO("best (3).pt")
print("=== AI 모델 로딩 완료! ===")

def run_ai_simulator():
    # [수정 포인트 1] 시뮬레이션 시작 전 서버 연결 상태를 단 한번 먼저 체크
    print("=== 백엔드 서버 연결 상태 확인 중... ===")
    is_server_online = True
    
    try:
        # 루트 경로("/")로 가볍게 요청을 보내 서버 생존 여부 확인 (타임아웃 2초)
        response = requests.get("http://127.0.0.1:8000/", timeout=2.0)
        if response.status_code != 200:
            is_server_online = False
    except requests.exceptions.ConnectionError:
        is_server_online = False

    # 서버가 안 켜져있을 때 딱 한 번 로그를 띄우고 3초 대기
    if not is_server_online:
        print("[오류] 백엔드 서버와 연결할 수 없습니다.")
        print("⏳ 3초 후 여기 코드 안에서만 독자적으로 시뮬레이션이 동작합니다...")
        time.sleep(3)
    else:
        print("[성공] 백엔드 서버와 정상적으로 연결되었습니다!")

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

                # 바운딩 박스가 그려진 이미지(프레임) 생성
                annotated_frame = r.plot()

                # 창 크기를 임의로 조절할 수 있도록 설정
                cv2.namedWindow("YOLO Traffic Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLO Traffic Detection", 800, 450)  # 창 크기 조절

                # 컴퓨터 화면에 팝업 창으로 시각화해서 띄우기
                cv2.imshow("YOLO Traffic Detection", annotated_frame)
                cv2.waitKey(1) # 창이 꺼지지 않고 부드럽게 갱신되도록 대기

            payload = {
                "pedestrian_count": 0,
                "exit_congested": (car_count > 15),
                "accident_detected": False,
                "emergency_detected": False,
                "north_cars": car_count,
                "south_cars": max(2, car_count // 2)
            }
            
            # 서버가 켜져있을 때만 POST 요청 시도
            if is_server_online:
                try:
                    response = requests.post(SERVER_URL, json=payload, timeout=2.0)
                    if response.status_code == 200:
                        res_json = response.json()
                        print(f"[전송 성공] 서버 응답 결과:")
                        print(f"  - 감지된 차량 수(AI): {car_count}대")
                        print(f"  - 제어 모드: {res_json.get('mode')}")
                        print(f"  - 신호 액션: {res_json.get('signal_action')}")
                    else:
                        print(f"[전송 실패] 상태 코드: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    # 실행 중에 서버가 갑자기 꺼진 경우 처리
                    is_server_online = False
                    print("[오류] 서버와의 연결이 끊어졌습니다. 지금부터 로컬 모드로 전환됩니다.")
            else:
                # 서버가 애초에 없거나 꺼져있어서 로컬 코드 안에서만 도는 상황
                print(f"[로컬 모드 동작] 감지된 차량 수(AI): {car_count}대 (서버 미연동)")

        except Exception as e:
            print(f"[오류 발생]: {e}")
            time.sleep(1)

        # 시연할 때 눈으로 확인하기 좋게 1초 대기
        time.sleep(1)

    cap.release()

if __name__ == "__main__":
    run_ai_simulator()