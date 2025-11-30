import json
import os
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
    "identity": "SummarizerAgent",
    "capabilities": ["summarizeText"],
    "schemas": {
        "summarizeText": {
            "input": {"text": "string"},
            "output": {"summary": "string"}
        }
    },
    "endpoint": "http://localhost:8000/api",
    "auth_methods": ["none"],  # 프로덕션에서는 OAuth2, API 키 등을 사용하세요
    "version": "1.0"
}


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/.well-known/agent.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(agent_card, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            rpc_request = json.loads(post_data)
            
            # JSON-RPC 요청 처리 (A2A의 핵심)
            if rpc_request.get('jsonrpc') == '2.0' and rpc_request['method'] == 'summarizeText':
                text = rpc_request['params']['text']
                
                # OpenAI API를 사용한 실제 LLM 요약
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                try:
                    llm_response = client.chat.completions.create(
                        model="gpt-5-mini",
                        messages=[
                            {"role": "system", "content": "당신은 간결한 요약을 제공하는 유용한 어시스턴트입니다."},
                            {"role": "user", "content": f"다음 텍스트를 요약하세요:\n{text}"}
                        ],
                    )
                    summary = llm_response.choices[0].message.content.strip()
                except Exception as e:
                    summary = f"요약 중 오류 발생: {str(e)}"
                
                response = {
                    "jsonrpc": "2.0",
                    "result": {"summary": summary},
                    "id": rpc_request['id']
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            else:
                # JSON-RPC에 따른 오류 처리
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "메서드를 찾을 수 없음"},
                    "id": rpc_request.get('id')
                }
                self.send_response(400)
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
    httpd.serve_forever()