# Chapter 04: 도구 사용 (Tool Use) 및 MCP

이 디렉터리는 LLM이 외부 도구를 사용하는 방법과 MCP(Model Context Protocol)를 활용하는 예제를 다룹니다.

## 파일 목록

*   `calculator_tool_use.py`: 사칙연산 도구를 정의하고 LLM이 이를 활용하여 계산 문제를 해결하는 예제입니다.
*   `wikipedia_tool_use.py`: 위키피디아 검색 도구를 사용하여 정보를 조회하는 예제입니다.
*   `stock_price_tool_use.py`: 주식 가격 정보를 조회하는 도구 사용 예제입니다.
*   `langgraph_mcp_client.py`: LangGraph 기반 에이전트가 MCP 서버(수학, 날씨)를 도구로 활용하는 클라이언트 예제입니다.
*   `src/common/mcp/MCP_math_server.py` (실제 위치: `mcp/MCP_math_server.py`): 수학 연산을 처리하는 MCP 서버입니다.
*   `src/common/mcp/MCP_weather_server.py` (실제 위치: `mcp/MCP_weather_server.py`): 날씨 정보를 제공하는 MCP 서버입니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  각 예제 스크립트를 실행합니다.

```bash
# 계산기 도구 예제
python calculator_tool_use.py

# 위키피디아 도구 예제
python wikipedia_tool_use.py

# 주식 가격 도구 예제
python stock_price_tool_use.py

# LangGraph + MCP 클라이언트 예제
# 주의: mcp/MCP_weather_server.py 등 서버를 별도 터미널에서 실행해야 할 수 있습니다.
python langgraph_mcp_client.py
```
