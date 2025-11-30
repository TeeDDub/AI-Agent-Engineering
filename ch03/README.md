# Chapter 03: 실시간 음성 에이전트

이 디렉터리는 OpenAI Realtime API와 WebSockets를 사용하여 웹 브라우저와 실시간으로 음성 대화를 나누는 에이전트 예제를 포함합니다.

## 파일 목록

*   `realtime_voice_agent.py`: FastAPI와 WebSockets를 이용한 백엔드 서버입니다. OpenAI Realtime API와 연결하여 음성 데이터를 중계합니다.
*   `index.html`: 웹 브라우저에서 실행되는 클라이언트입니다. 마이크 입력을 서버로 전송하고, 서버에서 받은 음성을 재생합니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  다음 명령어로 서버를 실행합니다.

```bash
python realtime_voice_agent.py
```
(또는 `uvicorn realtime_voice_agent:app --host 0.0.0.0 --port 5050`)

3.  서버가 실행되면 웹 브라우저에서 `index.html` 파일을 엽니다. (로컬 파일로 열거나 간단한 HTTP 서버를 띄워서 접속 가능)
