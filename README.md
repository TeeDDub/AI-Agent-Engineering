# AI 에이전트 엔지니어링

한빛미디어에서 출간된 \[AI 에이전트 엔지니어링](원제: Building Applications with AI Agents)의 예시 코드 저장소입니다.

## 예시 코드

이 책의 예제 코드는 깃허브에서 다운로드 할 수 있습니다. 한국어판 예제 코드는 업데이트를 반영하여 Python 3.12와 LangChain 1.0 환경에 맞게 수정하고, 장별로 정리했습니다. 예제에서 사용하는 일부 의존성의 특성상 윈도에서는 WSL 환경에서 실행하실 것을 권장합니다. 원서의 코드는 `/src`와 `/tests` 디렉터리에서 찾을 수 있습니다.

- 한국어판 저장소: https://github.com/TeeDDub/AI-Agent-Engineering
- 원서 저장소: https://github.com/michaelalbada/BuildingApplicationsWithAIAgents

## 환경 설정

이 프로젝트는 패키지 매니저로 [uv](https://github.com/astral-sh/uv)를 사용합니다.

### 1. uv 설치

`uv`가 설치되어 있지 않다면 공식 문서를 참고하여 설치해주세요.

```bash
# MacOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 프로젝트 설정

프로젝트 루트에서 다음 명령어를 실행하여 가상환경을 생성하고 의존성을 설치합니다.

```bash
uv sync
```

이 명령어는 `.venv` 디렉토리에 가상환경을 생성하고, `uv.lock`에 정의된 패키지들을 설치합니다.

### 3. 가상환경 활성화

```bash
# MacOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```


