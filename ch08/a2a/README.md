# Agent-to-Agent (A2A) Communication

이 디렉터리는 에이전트 간 직접 통신(Agent-to-Agent Communication)을 구현하는 예제 코드를 포함합니다. 표준화된 프로토콜(JSON-RPC over HTTP)을 사용하여 에이전트가 서로를 발견하고, 기능을 확인하며, 작업을 요청하는 과정을 보여줍니다.

## 파일 목록

*   `agent_server.py`: 요약 기능(`summarizeText`)을 제공하는 에이전트 서버입니다. HTTP 서버를 실행하고, `/.well-known/agent.json` 엔드포인트를 통해 자신의 기능(Agent Card)을 노출하며, `/api` 엔드포인트에서 JSON-RPC 요청을 처리합니다.
*   `agent_client.py`: 서버 에이전트를 발견하고 작업을 요청하는 클라이언트 에이전트(또는 스크립트)입니다. 에이전트 카드를 조회하여 호환성을 확인한 후, 텍스트 요약 작업을 요청합니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  두 개의 터미널을 엽니다.

### 터미널 1: 에이전트 서버 실행

서버를 먼저 실행하여 요청을 받을 준비를 합니다.

```bash
python agent_server.py
```
서버는 `http://localhost:8000`에서 실행됩니다.

### 터미널 2: 클라이언트 실행

클라이언트를 실행하여 서버에 작업을 요청합니다.

```bash
python agent_client.py
```

성공적으로 실행되면, 클라이언트는 서버를 발견하고("핸드셰이크 성공"), 텍스트 요약 요청을 보낸 후 결과를 출력합니다.










