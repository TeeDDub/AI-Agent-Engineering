# ADAS (Automated Design of Agentic Systems)

이 디렉터리는 에이전트 시스템을 자동으로 설계하고 최적화하는 연구적 접근 방식인 ADAS(Automated Design of Agentic Systems) 관련 코드를 포함합니다. Meta-Agent Search를 통해 새로운 에이전트 아키텍처를 발견하고 성능을 평가합니다.

## 파일 목록

*   `automated_design_of_agentic_systems.py`: ADAS 프레임워크의 메인 진입점입니다. MMLU(객관식 문제)와 ARC(추상적 추론) 태스크를 지원하며, 에이전트를 생성, 평가, 개선하는 루프를 실행합니다.
*   `mmlu_prompt.py`: MMLU 태스크를 위한 초기 프롬프트, 아카이브, 반사(Reflexion) 프롬프트 등을 제공합니다.
*   `arc_prompt.py`: ARC 태스크를 위한 초기 프롬프트, 아카이브 등을 제공합니다.
*   `utils.py`: 데이터 포맷팅, 평가 함수 등 유틸리티 함수들을 포함합니다.
*   `sample_mmlu_data.csv`: MMLU 테스트를 위한 샘플 데이터입니다.
*   `arc_val.pkl`, `arc_test.pkl`: ARC 태스크를 위한 검증 및 테스트 데이터셋입니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  태스크(MMLU 또는 ARC)에 따라 아래 명령어로 실행합니다.

### MMLU 태스크 실행

```bash
uv run python automated_design_of_agentic_systems.py \
    --task mmlu \
    --data_filename sample_mmlu_data.csv \
    --valid_size 2 \
    --test_size 1 \
    --n_generation 1 \
    --save_dir results
```

### ARC 태스크 실행

```bash
uv run python automated_design_of_agentic_systems.py \
    --task arc \
    --val_data_path arc_val.pkl \
    --test_data_path arc_test.pkl \
    --n_generation 2 \
    --save_dir results
```

실행 결과(발견된 에이전트 아키텍처 및 성능)는 `results/` 디렉터리에 저장됩니다.
