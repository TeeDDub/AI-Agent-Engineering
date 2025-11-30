# Chapter 06: 메모리 (Memory)

이 디렉터리는 에이전트에 기억(Memory) 능력을 부여하는 다양한 기법을 다룹니다.

## 파일 목록

*   `short_term_memory.py`: LangGraph의 `MemorySaver`를 사용하여 대화의 맥락(State)을 유지하고 단기 기억을 구현하는 방법을 보여줍니다.
*   `basic_bm25.py`: BM25 알고리즘을 사용하여 텍스트 말뭉치에서 키워드 기반 검색을 수행하는 기본적인 예제입니다. (장기 기억 검색의 기초)

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  각 예제 스크립트를 실행합니다.

```bash
# 단기 기억(Checkpointer) 예제
python short_term_memory.py

# BM25 검색 예제
python basic_bm25.py
```
