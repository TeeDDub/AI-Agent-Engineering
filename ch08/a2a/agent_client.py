import requests
import json

# 단계 1: 레지스트리 또는 알려진 URL을 통해 에이전트 검색 (직접 접근으로 모킹)
card_url = 'http://localhost:8000/.well-known/agent.json'
response = requests.get(card_url)
if response.status_code != 200:
    print("에이전트 카드 가져오기 실패")
    exit()

agent_card = response.json()
print("발견된 에이전트 카드:", json.dumps(agent_card, indent=2, ensure_ascii=False))

# 단계 2: 핸드셰이크 (모킹: 버전 및 기능 확인)
if agent_card['version'] != '1.0':
    print("호환되지 않는 프로토콜 버전")
    exit()
if "summarizeText" not in agent_card['capabilities']:
    print("필요한 기능이 지원되지 않음")
    exit()
print("핸드셰이크 성공: 에이전트가 호환됩니다.")

# 단계 3: 구조화된 JSON-RPC 요청 발행
rpc_url = agent_card['endpoint']
rpc_request = {
    "jsonrpc": "2.0",
    "method": "summarizeText",
    "params": {"text": "이것은 요약이 필요한 긴 예제 텍스트입니다. 멀티 에이전트 시스템, 통신 프로토콜, 그리고 A2A와 같은 표준을 사용하여 에이전트들이 어떻게 자율적으로 협업할 수 있는지에 대해 논의합니다."},
    "id": 123  # 고유한 요청 ID
}

response = requests.post(rpc_url, json=rpc_request)
if response.status_code == 200:
    rpc_response = response.json()
    print("RPC 응답:", json.dumps(rpc_response, indent=2, ensure_ascii=False))
else:
    print("오류:", response.status_code, response.text)
