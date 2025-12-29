import requests
import json
import uuid

# 단계 1: 에이전트 검색 - A2A 스펙 준수 well-known URI
card_url = 'http://localhost:8000/.well-known/agent-card.json'
response = requests.get(card_url)
if response.status_code != 200:
    print("에이전트 카드 가져오기 실패")
    exit()

agent_card = response.json()
print("발견된 에이전트 카드:", json.dumps(agent_card, indent=2, ensure_ascii=False))

# 단계 2: 핸드셰이크 (버전 및 기능 확인)
if agent_card.get('protocolVersion', '').split('.')[0] != '1':
    print("호환되지 않는 프로토콜 버전")
    exit()

# skills 확인
skills = agent_card.get('skills', [])
skill_ids = [s.get('id') for s in skills]
if "summarize-text" not in skill_ids:
    print("필요한 스킬이 지원되지 않음")
    exit()
print("핸드셰이크 성공: 에이전트가 호환됩니다.")

# 단계 3: A2A 스펙 준수 JSON-RPC 요청 (message/send)
rpc_url = agent_card['url']  # 에이전트 기본 URL로 POST
rpc_request = {
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
        "contextId": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [
                {
                    "text": "이것은 요약이 필요한 긴 예제 텍스트입니다. 멀티 에이전트 시스템, 통신 프로토콜, A2A와 같은 표준을 사용하여 에이전트들이 자율적으로 협업하는 방법을 논의합니다."
                }
            ]
        }
    },
    "id": 123  # 고유한 요청 ID
}

response = requests.post(rpc_url, json=rpc_request)
if response.status_code == 200:
    rpc_response = response.json()
    print("\nRPC 응답:", json.dumps(rpc_response, indent=2, ensure_ascii=False))
    
    # 결과 파싱
    if 'result' in rpc_response:
        result = rpc_response['result']
        print(f"\n태스크 ID: {result.get('id')}")
        print(f"상태: {result.get('status', {}).get('state')}")
        
        artifacts = result.get('artifacts', [])
        if artifacts:
            for artifact in artifacts:
                for part in artifact.get('parts', []):
                    if 'text' in part:
                        print(f"\n요약 결과:\n{part['text']}")
else:
    print("오류:", response.status_code, response.text)
