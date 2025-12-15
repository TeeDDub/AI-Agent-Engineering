# Chapter 07: 에이전트 학습 및 최적화

이 디렉터리는 에이전트가 경험을 통해 학습하고 성능을 최적화하는 다양한 기법(Reflexion, 파인튜닝, 강화학습 등)을 다룹니다.

## 파일 목록

*   `reflexion.py`: 에이전트가 실패한 작업을 성찰(Reflection)하고, 이를 메모리에 저장하여 다음 시도에서 성능을 개선하는 Reflexion 패턴 구현 예제입니다.
*   `reflexion_trial_learning.py`: 여러 번의 시도(Trial)를 통해 에이전트가 스스로 학습하고 전략을 수정하는 과정을 보여줍니다.
*   `experiential_learning.py`: 에이전트가 경험(성공/실패 사례)을 통해 인사이트를 생성하고, 이를 승격/강등하며 지식을 축적하는 과정을 시뮬레이션합니다.
*   `supervised_fine_tuning.py`: 지도 학습(Supervised Fine-Tuning, SFT)을 통해 모델을 특정 작업에 맞게 미세 조정하는 예제입니다.
*   `direct_preference_optimization.py`: DPO(Direct Preference Optimization)를 사용하여 인간의 선호도에 맞춰 모델을 최적화하는 예제입니다.
*   `reinforcement_learning_with_verifiable_rewards.py`: 검증 가능한 보상(Verifiable Rewards)을 기반으로 강화 학습(RL)을 수행하여 에이전트의 도구 사용 및 형식 준수 능력을 향상시키는 예제입니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  각 예제 스크립트를 실행합니다. (파인튜닝 관련 스크립트는 GPU 자원 및 데이터셋 준비가 필요할 수 있습니다.)

```bash
# Reflexion 예제
python reflexion.py

# 경험적 학습 예제
python experiential_learning.py
```



