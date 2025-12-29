import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from openai import OpenAI

# 환경변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 에이전트 카드 (발견을 위한 JSON 설명자)
agent_card = {
    "name": "SummarizerAgent",
    "description": "텍스트 요약을 수행하는 AI 에이전트입니다.",
    "protocolVersion": "1.0",
    "url": "http://localhost:8000",
    "provider": {
        "organization": "Example Org",
        "url": "https://example.org"
    },
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False
    },
    "skills": [
        {
            "id": "summarize-text",
            "name": "텍스트 요약",
            "description": "주어진 텍스트를 간결하게 요약합니다.",
            "tags": ["summarization", "nlp", "text-processing"],
            "examples": [
                "이 기사를 요약해 주세요",
                "다음 내용을 간단히 정리해 주세요"
            ]
        }
    ],
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "security": []  # 프로덕션에서는 OAuth2, API 키 등을 설정하세요
}


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # A2A 스펙: /.well-known/agent-card.json (섹션 8.2, 14.3)
        if self.path == '/.well-known/agent-card.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(agent_card, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            rpc_request = json.loads(post_data)
            
            # A2A JSON-RPC 요청 처리 (섹션 9.4.1 message/send)
            if rpc_request.get('jsonrpc') == '2.0' and rpc_request['method'] == 'message/send':
                params = rpc_request.get('params', {})
                message = params.get('message', {})
                parts = message.get('parts', [])
                
                # 텍스트 파트 추출 (섹션 4.1.6 Part)
                text = ""
                for part in parts:
                    if 'text' in part:
                        text += part['text']
                
                # OpenAI API를 사용한 실제 LLM 요약
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                try:
                    llm_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "당신은 간결한 요약을 제공하는 유용한 어시스턴트입니다."},
                            {"role": "user", "content": f"다음 텍스트를 요약하세요:\n{text}"}
                        ],
                    )
                    summary = llm_response.choices[0].message.content.strip()
                except Exception as e:
                    summary = f"요약 중 오류 발생: {str(e)}"
                
                # A2A 스펙 준수 응답 (섹션 4.1.1 Task, 4.1.2 TaskStatus)
                task_id = str(uuid.uuid4())
                response = {
                    "jsonrpc": "2.0",
                    "result": {
                        "id": task_id,
                        "contextId": params.get('contextId', str(uuid.uuid4())),
                        "status": {
                            "state": "completed"
                        },
                        "artifacts": [
                            {
                                "parts": [{"text": summary}]
                            }
                        ]
                    },
                    "id": rpc_request['id']
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            else:
                # JSON-RPC 오류 처리 (섹션 9.5)
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": rpc_request.get('id')
                }
                self.send_response(200)  # JSON-RPC 에러도 HTTP 200으로 반환
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, AgentHandler)
    print("A2A 에이전트 서버를 시작합니다. 주소: http://localhost:8000")
    print("Agent Card: http://localhost:8000/.well-known/agent-card.json")
    httpd.serve_forever()