# Chapter 12: 프롬프트 인젝션 방어

이 디렉터리는 LLM 에이전트에 대한 프롬프트 인젝션 공격을 탐지하고 방어하는 기법을 다룹니다.

## 파일 목록

*   `prompt_injection_block.py`: `llm-guard` 라이브러리를 사용하여 입력 프롬프트에서 개인정보(PII)를 익명화하고, 악의적인 부분 문자열을 차단하며, 프롬프트 인젝션 시도를 방어하는 예제입니다.

## 실행 방법

1.  필요한 라이브러리(`llm-guard`)가 설치되어 있어야 합니다.
2.  스크립트 실행:

```bash
python prompt_injection_block.py
```









