import time
import requests

# FastAPI 백엔드 서버 주소 (ngrok 주소 또는 로컬 IP)
SERVER_URL = "http://127.0.0.1:8000/api/control/"

def send_traffic_data():
    print("=== AI 분석 및 전송 시뮬레이터 시작 ===")
    
    # 테스트할 주요 시나리오 리스트업 (순차적으로 전환됨)
    scenarios = [
        {
            "name": "1. 평상시 (Normal)",
            "data": {"pedestrian_count": 1, "exit_congested": False, "accident_detected": False, "emergency_detected": False, "north_cars": 10, "south_cars": 5}
        },
        {
            "name": "2. 단독 응급 차량 진입",
            "data": {"pedestrian_count": 0, "exit_congested": False, "accident_detected": False, "emergency_detected": True, "north_cars": 5, "south_cars": 5}
        },
        {
            "name": "3. 단독 사고 발생",
            "data": {"pedestrian_count": 0, "exit_congested": False, "accident_detected": True, "emergency_detected": False, "north_cars": 5, "south_cars": 5}
        },
        {
            "name": "4. 출구 정체 (꼬리물기)",
            "data": {"pedestrian_count": 0, "exit_congested": True, "accident_detected": False, "emergency_detected": False, "north_cars": 12, "south_cars": 8}
        },
        {
            "name": "5. 복합 재난 상황 (응급차 + 사고 + 정체 동시 발생)",
            "data": {"pedestrian_count": 6, "exit_congested": True, "accident_detected": True, "emergency_detected": True, "north_cars": 15, "south_cars": 15}
        }
    ]

    index = 0
    while True:
        current_scenario = scenarios[index]
        current_status = current_scenario["data"]
        
        try:
            print(f"\n--------------------------------------------------")
            print(f"[시뮬레이터] 현재 테스트 상황: {current_scenario['name']}")
            
            # 서버로 데이터 POST 요청 전송
            response = requests.post(SERVER_URL, json=current_status)
            
            if response.status_code == 200:
                res_json = response.json()
                print(f"[전송 성공] 서버 응답 결과:")
                print(f"  - 제어 모드 (Mode): {res_json.get('mode')}")
                print(f"  - 감지된 이벤트 목록 (Active Events): {res_json.get('active_events')}")
                print(f"  - 안내 메시지 (Message): {res_json.get('message')}")
            else:
                print(f"[전송 실패] 상태 코드: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("[오류] 서버와 연결할 수 없습니다. 백엔드 주소나 네트워크를 확인하세요.")

        # 다음 시나리오로 인덱스 순환 (5초 대기)
        index = (index + 1) % len(scenarios)
        time.sleep(5)

if __name__ == "__main__":
    send_traffic_data()