# Chapter 11: 보안 운영 센터 (SOC) 에이전트

이 디렉터리는 보안 위협을 탐지, 분석 및 대응하는 SOC 분석가 에이전트와 이를 최적화하는 기법을 다룹니다.

## 파일 목록

*   `soc_analyst_agent.py`: 위협 인텔리전스 조회, 로그 분석, 사고 분류, 호스트 격리 등의 도구를 갖춘 SOC 분석가 에이전트의 LangGraph 구현체입니다.
*   `optimize_soc_react_agent.py`: DSPy를 사용하여 SOC 에이전트(ReAct 패턴)의 프롬프트와 로직을 최적화하는 예제입니다.
*   `optimize_threat_classifier.py`: DSPy를 사용하여 위협 분류기(Classifier)의 성능을 최적화하는 예제입니다. 합성 데이터를 생성하고 이를 바탕으로 분류 모델을 튜닝합니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  SOC 에이전트 실행:

```bash
python soc_analyst_agent.py
```

3.  DSPy 최적화 예제 실행:
```bash
python optimize_soc_react_agent.py
python optimize_threat_classifier.py
```

