# Chapter 08: 멀티 에이전트 공급망 관리

이 디렉터리는 복잡한 공급망 및 물류 관리를 위해 여러 에이전트가 협력하는 시스템을 구축하는 방법을 다룹니다.

## 파일 목록

*   `supply_chain_logistics_agent.py`: LangGraph를 사용하여 재고, 운송, 공급업체 관리 등 다양한 도구를 활용하는 단일/멀티 에이전트 워크플로우의 기본 구현입니다.
*   `ray_supply_chain_multi_agent.py`: Ray 프레임워크를 사용하여 분산 환경에서 멀티 에이전트 공급망 시스템을 구동하는 예제입니다.
*   `temporal_supply_chain_multi_agent.py`: Temporal 워크플로우 엔진을 사용하여 내구성 있고 신뢰성 있는 멀티 에이전트 오케스트레이션을 구현한 예제입니다.
*   `adas/automated_design_of_agentic_systems.py`: 에이전트 시스템을 자동 설계(Automated Design of Agentic Systems, ADAS)하는 연구적 접근 방식을 보여주는 코드입니다.
*   `a2a/agent_server.py`: Agent-to-Agent 통신을 위한 서버 측 구현입니다.
*   `a2a/agent_client.py`: Agent-to-Agent 통신을 위한 클라이언트 측 구현입니다.

## 실행 방법

1.  프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정합니다.
2.  Loki, Ray, Temporal 등 추가적인 인프라가 필요한 예제는 해당 서비스를 먼저 실행해야 합니다.
3.  기본 에이전트 실행:

```bash
python supply_chain_logistics_agent.py
```

4.  Ray 기반 멀티 에이전트 실행 (Ray 설치 필요):
```bash
python ray_supply_chain_multi_agent.py
```

5.  Temporal 기반 실행 (Temporal 서버 실행 필요):
```bash
python temporal_supply_chain_multi_agent.py
```
