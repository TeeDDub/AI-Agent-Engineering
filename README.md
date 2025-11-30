# Building Applications with AI Agents

"Building Applications with AI Agents" 도서의 예시 코드 저장소입니다.

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


