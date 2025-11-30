# Chapter 05: 스킬 선택 및 에이전트 아키텍처

이 디렉터리는 LLM이 적절한 도구(스킬)를 선택하는 다양한 방법과 에이전트 아키텍처를 구성하는 예제를 포함합니다.

## 파일 목록

*   `basic_skill_selection.py`: LLM을 사용하여 쿼리에 적합한 스킬 그룹과 도구를 선택하는 기본적인 예제입니다.
*   `semantic_skill_selection.py`: 의미론적 검색(Semantic Search, 임베딩 + 벡터 DB)을 사용하여 사용자 쿼리와 유사도가 높은 도구를 선택하는 예제입니다.
*   `hierarchical_skill_selection.py`: 계층적 스킬 선택 방식을 구현한 예제입니다. 먼저 스킬 그룹을 선택하고, 그 후 그룹 내 구체적인 도구를 선택합니다.
*   `langgraph_example.py`: LangGraph를 사용하여 분류, 처리, 라우팅 등 복잡한 워크플로우를 가진 에이전트를 구현하는 예제입니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  각 예제 스크립트를 실행합니다.

```bash
# 기본 스킬 선택 예제
python basic_skill_selection.py

# 의미론적 스킬 선택 예제
python semantic_skill_selection.py

# 계층적 스킬 선택 예제
python hierarchical_skill_selection.py

# LangGraph 에이전트 예제
python langgraph_example.py
```
