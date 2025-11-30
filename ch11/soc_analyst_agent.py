from __future__ import annotations
"""
soc_analyst_agent.py
위협 조사, 사고 대응, 로그 분석 및 보안 모니터링을 처리하는 보안 운영 센터(SOC) 분석가 에이전트를 위한 LangGraph 워크플로우입니다.
"""
import os
import json
import operator
import builtins
from typing import Annotated, Sequence, TypedDict, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from traceloop.sdk import Traceloop
from src.common.observability.loki_logger import log_to_loki

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"

@tool
def lookup_threat_intel(indicator: str, type: str, **kwargs) -> str:
    """IP 주소, 파일 해시, URL 및 도메인에 대한 위협 인텔리전스를 조회합니다."""
    print(f"[TOOL] lookup_threat_intel(indicator={indicator}, type={type}, kwargs={kwargs})")
    log_to_loki("tool.lookup_threat_intel", f"indicator={indicator}, type={type}")
    return "threat_intel_retrieved"

@tool
def query_logs(query: str, log_index: str, **kwargs) -> str:
    """인증, 엔드포인트, 네트워크, 방화벽 및 DNS 시스템 전반에 걸쳐 보안 로그를 검색하고 분석합니다."""
    print(f"[TOOL] query_logs(query={query}, log_index={log_index}, kwargs={kwargs})")
    log_to_loki("tool.query_logs", f"query={query}, log_index={log_index}")
    return "log_query_executed"

@tool
def triage_incident(incident_id: str, decision: str, reason: str, **kwargs) -> str:
    """보안 사고를 실제 긍정(True Positive), 오탐지(False Positive) 또는 추가 조사를 위한 에스컬레이션으로 분류합니다."""
    print(f"[TOOL] triage_incident(incident_id={incident_id}, decision={decision}, reason={reason}, kwargs={kwargs})")
    log_to_loki("tool.triage_incident", f"incident_id={incident_id}, decision={decision}")
    return "incident_triaged"

@tool
def isolate_host(host_id: str, reason: str, **kwargs) -> str:
    """측면 이동을 방지하고 보안 사고를 억제하기 위해 손상된 호스트를 격리합니다."""
    print(f"[TOOL] isolate_host(host_id={host_id}, reason={reason}, kwargs={kwargs})")
    log_to_loki("tool.isolate_host", f"host_id={host_id}, reason={reason}")
    return "host_isolated"

@tool
def send_analyst_response(incident_id: str = None, message: str = None) -> str:
    """보안 분석, 사고 업데이트 또는 권장 사항을 이해 관계자에게 전송합니다."""
    print(f"[TOOL] send_analyst_response → {message}")
    log_to_loki("tool.send_analyst_response", f"incident_id={incident_id}, message={message}")
    return "analyst_response_sent"

TOOLS = [
    lookup_threat_intel, query_logs, triage_incident, isolate_host, send_analyst_response
]

llm = init_chat_model(model="gpt-5-mini", callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    incident: Optional[dict]  # 보안 사고 정보
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # 누락되거나 불완전한 사고 데이터를 우아하게 처리
    incident = state.get("incident", {})
    if not incident:
        incident = {"incident_id": "UNKNOWN", "severity": "medium", "status": "investigating", "analyst": "SOC_EVAL"}
    
    incident_json = json.dumps(incident, ensure_ascii=False)
    system_prompt = (
        "당신은 사이버 보안 사고 대응을 전문으로 하는 숙련된 보안 운영 센터(SOC) 분석가입니다.\n"
        "당신의 전문 분야는 다음과 같습니다:\n"
        "- 위협 인텔리전스 분석 및 침해 지표(IOC) 조사\n"
        "- 여러 시스템에 걸친 보안 로그 분석 및 상관관계 분석\n"
        "- 사고 분류 및 선별(triage) (실제 긍정/오탐지)\n"
        "- 악성코드 분석 및 위협 사냥(Threat Hunting)\n"
        "- 네트워크 보안 모니터링 및 이상 징후 탐지\n"
        "- 사고 봉쇄 및 대응 조정\n"
        "- SIEM/SOAR 플랫폼 운영\n"
        "\n"
        "조사 방법론:\n"
        "  1) 보안 경고를 분석하고 초기 지표를 수집합니다.\n"
        "  2) lookup_threat_intel을 사용하여 IP, 해시, URL 및 도메인을 조사합니다.\n"
        "  3) query_logs를 사용하여 증거를 찾기 위해 관련 로그 소스를 검색합니다.\n"
        "  4) triage_incident를 사용하여 결과를 실제 긍정/오탐지로 분류합니다.\n"
        "  5) 확산을 방지하기 위해 격리가 필요한 경우 isolate_host를 사용합니다.\n"
        "  6) 결과를 문서화하기 위해 send_analyst_response로 후속 조치를 취합니다.\n"
        "\n"
        "항상 신속한 위협 봉쇄와 정확한 사고 분류를 최우선으로 하세요.\n\n"
        f"사고(INCIDENT): {incident_json}"
    )

    full = [SystemMessage(content=system_prompt)] + history

    first: ToolMessage | BaseMessage = llm.invoke(full)
    messages = [first]

    if getattr(first, "tool_calls", None):
        for tc in first.tool_calls:
            print(first)
            print(tc['name'])
            fn = next(t for t in TOOLS if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

        second = llm.invoke(full + messages)
        messages.append(second)

    return {"messages": messages}

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    Traceloop.init(disable_batch=True, app_name="soc_analyst_agent")
    example = {"incident_id": "INC-12345", "severity": "high", "type": "suspicious_login", "analyst": "J.Smith"}
    convo = [HumanMessage(content='경고가 발생했습니다: "IP 203.0.113.45에서 관리자 계정으로 의심스러운 로그인 시도." 어떻게 해야 하나요?')]
    result = graph.invoke({"incident": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}") 