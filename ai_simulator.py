import time
import requests

# FastAPI 백엔드 서버 주소 (ngrok 주소 또는 로컬 IP)
SERVER_URL = "http://127.0.0.1:8000/api/control/"

def send_traffic_data():
    print("=== AI 분석 및 전송 시뮬레이터 시작 ===")
    
    # 초기 상태 설정 (정상)
    current_status = {
        "pedestrian_count": 2,
        "exit_congested": False,
        "accident_detected": False,
        "emergency_detected": False,
        "north_cars": 10,
        "south_cars": 12
    }

    while True:
        try:
            # 서버로 데이터 POST 요청 전송
            response = requests.post(SERVER_URL, json=current_status)
            
            if response.status_code == 200:
                print(f"[전송 성공] accident={current_status['accident_detected']}, emergency={current_status['emergency_detected']}")
                print(f"[서버 응답]: {response.json()}\n")
            else:
                print(f"[전송 실패] 상태 코드: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("[오류] 서버와 연결할 수 없습니다. 백엔드 주소나 네트워크를 확인하세요.")

        # 3초 대기 후 반복
        time.sleep(3)

if __name__ == "__main__":
    send_traffic_data()