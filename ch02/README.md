# Chapter 02: 간단한 에이전트 구축 및 평가

이 디렉터리는 간단한 고객 지원 에이전트를 구축하고 평가하는 예제 코드를 포함합니다.

## 파일 목록

*   `simple_customer_support_agent.py`: LangGraph와 LangChain을 사용하여 구축된 간단한 주문 취소 지원 에이전트입니다.
*   `customer_support_agent_evaluation.py`: 위 에이전트가 정상적으로 동작하는지(도구 호출, 응답 메시지 등) 평가하는 스크립트입니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  다음 명령어로 에이전트를 실행하거나 평가할 수 있습니다.

```bash
# 에이전트 실행
python simple_customer_support_agent.py

# 에이전트 평가
python customer_support_agent_evaluation.py
```









