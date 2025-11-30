from __future__ import annotations
"""
supply_chain_logistics_agent_ray_per_session.py
세션별 격리를 통한 Ray 액터를 사용하는 멀티 에이전트 공급망 및 물류 관리 시스템을 위한 LangGraph 워크플로우.
Ray 액터로 구성된 전문 에이전트를 통해 재고 관리, 운송 작업, 공급업체 관계 및 창고 최적화를 처리합니다.
각 세션(operation_id로 식별)은 전문가 유형별로 자체 액터 인스턴스를 가지며, 세션별 격리된 상태와 순차 실행을 보장합니다.
슈퍼바이저는 전문가를 결정하고 SessionManager를 통해 세션별 액터를 원격으로 호출합니다.
실행법: 파이썬 3.12를 기준으로 생성한 venv에서 pip로 ray를 설치하고, python ch08/ray_supply_chain_multi_agent.py 명령어로 실행합니다.
"""

import os
import json
import time
from typing import Annotated, Sequence, TypedDict, Optional, Dict

import ray
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from traceloop.sdk import Traceloop
from src.common.observability.loki_logger import log_to_loki

# 환경변수 확인
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"

# 모든 전문가를 위한 공유 도구
@tool
def send_logistics_response(operation_id: Optional[str] = None, message: Optional[str] = None) -> str:
    """이해관계자에게 물류 업데이트, 권장 사항 또는 상태 보고서를 전송합니다."""
    print(f"[도구] send_logistics_response → {message}")
    log_to_loki("tool.send_logistics_response", f"operation_id={operation_id}, message={message}")
    return "logistics_response_sent"

# 재고 및 창고 전문가 도구
@tool
def manage_inventory(sku: Optional[str] = None, **kwargs) -> str:
    """재고 수준, 재고 보충, 감사 및 최적화 전략을 관리합니다."""
    print(f"[도구] manage_inventory(sku={sku}, kwargs={kwargs})")
    log_to_loki("tool.manage_inventory", f"sku={sku}")
    return "inventory_management_initiated"

@tool
def optimize_warehouse(operation_type: Optional[str] = None, **kwargs) -> str:
    """창고 운영, 레이아웃, 용량 및 보관 효율성을 최적화합니다."""
    print(f"[도구] optimize_warehouse(operation_type={operation_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_warehouse", f"operation_type={operation_type}")
    return "warehouse_optimization_initiated"

@tool
def forecast_demand(season: Optional[str] = None, **kwargs) -> str:
    """수요 패턴, 계절적 추세를 분석하고 예측 모델을 생성합니다."""
    print(f"[도구] forecast_demand(season={season}, kwargs={kwargs})")
    log_to_loki("tool.forecast_demand", f"season={season}")
    return "demand_forecast_generated"

@tool
def manage_quality(supplier: Optional[str] = None, **kwargs) -> str:
    """품질 관리, 결함 추적 및 공급업체 품질 표준을 관리합니다."""
    print(f"[도구] manage_quality(supplier={supplier}, kwargs={kwargs})")
    log_to_loki("tool.manage_quality", f"supplier={supplier}")
    return "quality_management_initiated"

@tool
def scale_operations(scaling_type: Optional[str] = None, **kwargs) -> str:
    """성수기, 용량 계획 및 인력 관리를 위한 운영을 확장합니다."""
    print(f"[도구] scale_operations(scaling_type={scaling_type}, kwargs={kwargs})")
    log_to_loki("tool.scale_operations", f"scaling_type={scaling_type}")
    return "operations_scaled"

@tool
def optimize_costs(cost_type: Optional[str] = None, **kwargs) -> str:
    """운송, 보관 및 운영 비용을 분석하고 최적화합니다."""
    print(f"[도구] optimize_costs(cost_type={cost_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_costs", f"cost_type={cost_type}")
    return "cost_optimization_initiated"

INVENTORY_TOOLS = [manage_inventory, optimize_warehouse, forecast_demand, manage_quality, scale_operations, optimize_costs, send_logistics_response]

# 운송 및 물류 전문가 도구
@tool
def track_shipments(origin: Optional[str] = None, **kwargs) -> str:
    """배송 상태, 지연 사항을 추적하고 배송 물류를 조정합니다."""
    print(f"[도구] track_shipments(origin={origin}, kwargs={kwargs})")
    log_to_loki("tool.track_shipments", f"origin={origin}")
    return "shipment_tracking_updated"

@tool
def arrange_shipping(shipping_type: Optional[str] = None, **kwargs) -> str:
    """배송 방법, 특급 배송 및 복합 운송을 준비합니다."""
    print(f"[도구] arrange_shipping(shipping_type={shipping_type}, kwargs={kwargs})")
    log_to_loki("tool.arrange_shipping", f"shipping_type={shipping_type}")
    return "shipping_arranged"

@tool
def coordinate_operations(operation_type: Optional[str] = None, **kwargs) -> str:
    """크로스도킹, 통합 및 이동과 같은 복잡한 작업을 조정합니다."""
    print(f"[도구] coordinate_operations(operation_type={operation_type}, kwargs={kwargs})")
    log_to_loki("tool.coordinate_operations", f"operation_type={operation_type}")
    return "operations_coordinated"

@tool
def manage_special_handling(product_type: Optional[str] = None, **kwargs) -> str:
    """위험물, 콜드체인 및 민감한 제품에 대한 특수 요구사항을 처리합니다."""
    print(f"[도구] manage_special_handling(product_type={product_type}, kwargs={kwargs})")
    log_to_loki("tool.manage_special_handling", f"product_type={product_type}")
    return "special_handling_managed"

@tool
def process_returns(returned_quantity: Optional[str] = None, **kwargs) -> str:
    """반품, 역물류 및 제품 처리를 처리합니다."""
    print(f"[도구] process_returns(returned_quantity={returned_quantity}, kwargs={kwargs})")
    log_to_loki("tool.process_returns", f"returned_quantity={returned_quantity}")
    return "returns_processed"

@tool
def optimize_delivery(delivery_type: Optional[str] = None, **kwargs) -> str:
    """배송 경로, 라스트마일 물류 및 지속가능성 이니셔티브를 최적화합니다."""
    print(f"[도구] optimize_delivery(delivery_type={delivery_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_delivery", f"delivery_type={delivery_type}")
    return "delivery_optimization_complete"

@tool
def manage_disruption(disruption_type: Optional[str] = None, **kwargs) -> str:
    """공급망 중단, 비상 계획 및 위험 완화를 관리합니다."""
    print(f"[도구] manage_disruption(disruption_type={disruption_type}, kwargs={kwargs})")
    log_to_loki("tool.manage_disruption", f"disruption_type={disruption_type}")
    return "disruption_managed"

TRANSPORTATION_TOOLS = [track_shipments, arrange_shipping, coordinate_operations, manage_special_handling, process_returns, optimize_delivery, manage_disruption, send_logistics_response]

# 공급업체 및 규정 준수 전문가 도구
@tool
def evaluate_suppliers(supplier_name: Optional[str] = None, **kwargs) -> str:
    """공급업체 성과를 평가하고 감사를 수행하며 공급업체 관계를 관리합니다."""
    print(f"[도구] evaluate_suppliers(supplier_name={supplier_name}, kwargs={kwargs})")
    log_to_loki("tool.evaluate_suppliers", f"supplier_name={supplier_name}")
    return "supplier_evaluation_complete"

@tool
def handle_compliance(compliance_type: Optional[str] = None, **kwargs) -> str:
    """규제 준수, 세관, 문서화 및 인증을 관리합니다."""
    print(f"[도구] handle_compliance(compliance_type={compliance_type}, kwargs={kwargs})")
    log_to_loki("tool.handle_compliance", f"compliance_type={compliance_type}")
    return "compliance_handled"

SUPPLIER_TOOLS = [evaluate_suppliers, handle_compliance, send_logistics_response]

# 메인 프로세스용 LLM (슈퍼바이저용)
llm = init_chat_model(model="gpt-5-mini", callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

class AgentState(TypedDict):
    operation: Optional[dict]  # 공급망 운영 정보
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 도구 이름으로 도구 맵 생성
TOOLS_MAP = {
    "inventory": INVENTORY_TOOLS,
    "transportation": TRANSPORTATION_TOOLS,
    "supplier": SUPPLIER_TOOLS,
}

# 전문가를 위한 Ray 액터 (세션별 격리)
# 주의: LLM 객체는 직렬화가 불가능하므로 액터 내부에서 생성해야 함
@ray.remote
class SpecialistActor:
    def __init__(self, name: str, tools_key: str, system_prompt: str):
        self.name = name
        # 액터 내부에서 LLM 초기화 (직렬화 문제 회피)
        base_llm = init_chat_model(model="gpt-5-mini", verbose=True)
        tools = TOOLS_MAP[tools_key]
        self.llm = base_llm.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        self.prompt = system_prompt
        self.internal_state = {}  # 세션별 격리된 상태, 예: 세션 내 추적용

    def process_task(self, operation: dict, messages: Sequence[BaseMessage]):
        if not operation:
            operation = {"operation_id": "알 수 없음", "type": "일반", "priority": "중간", "status": "활성"}
        operation_json = json.dumps(operation, ensure_ascii=False)
        full_prompt = self.prompt + f"\n\n작업: {operation_json}"
        
        full = [SystemMessage(content=full_prompt)] + messages

        first = self.llm.invoke(full)
        result_messages = [first]

        if hasattr(first, "tool_calls"):
            for tc in first.tool_calls:
                print(first)
                print(tc['name'])
                fn = self.tools.get(tc['name'])
                if fn:
                    out = fn.invoke(tc["args"])
                    result_messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

            second = self.llm.invoke(full + result_messages)
            result_messages.append(second)

        # 내부 상태 업데이트 (예: 세션 내에서 처리된 단계 추적)
        step_key = str(len(self.internal_state) + 1)  # 또는 더 구체적인 키 사용
        self.internal_state[step_key] = {"status": "처리됨", "timestamp": time.time()}

        return {"messages": result_messages}

    def get_state(self):
        return self.internal_state  # 전체 세션 상태 반환

# 세션 관리자 액터: 세션별 전문가 액터를 추적
@ray.remote
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, ray.actor.ActorHandle]] = {}  # session_id -> {agent_name: actor}

    def get_or_create_actor(self, session_id: str, agent_name: str, prompt: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        if agent_name not in self.sessions[session_id]:
            # LLM은 액터 내부에서 생성됨 (직렬화 불가능하므로)
            actor = SpecialistActor.remote(agent_name, agent_name, prompt)
            self.sessions[session_id][agent_name] = actor
        return self.sessions[session_id][agent_name]

    def get_session_state(self, session_id: str, agent_name: str):
        if session_id in self.sessions and agent_name in self.sessions[session_id]:
            actor = self.sessions[session_id][agent_name]
            return actor.get_state.remote()  # future 반환
        return None

# 슈퍼바이저: 전문가를 결정하고 관리자를 통해 세션별 Ray 액터를 원격으로 호출
def supervisor_invoke(operation: dict, messages: Sequence[BaseMessage], manager: ray.actor.ActorHandle, prompts: dict):
    session_id = operation.get("operation_id", "알 수 없음")
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    supervisor_prompt = (
        "당신은 공급망 전문가 팀을 조정하는 슈퍼바이저입니다.\n"
        "팀원:\n"
        "- inventory: 재고 수준, 예측, 품질, 창고 최적화, 확장 및 비용을 처리합니다.\n"
        "- transportation: 배송 추적, 준비, 운영 조정, 특수 처리, 반품, 배송 최적화 및 중단을 처리합니다.\n"
        "- supplier: 공급업체 평가 및 규정 준수를 처리합니다.\n"
        "\n"
        "사용자 쿼리를 기반으로 처리할 팀원 한 명을 선택하세요.\n"
        "선택한 팀원의 이름(inventory, transportation 또는 supplier)만 출력하고 다른 것은 출력하지 마세요.\n\n"
        f"작업: {operation_json}"
    )

    full = [SystemMessage(content=supervisor_prompt)] + messages
    response = llm.invoke(full)
    agent_name = response.content.strip().lower()
    
    if agent_name not in prompts:
        raise ValueError(f"알 수 없는 에이전트: {agent_name}")
    
    # 세션별 액터 가져오기 또는 생성 (LLM은 액터 내부에서 생성됨)
    actor_ref = manager.get_or_create_actor.remote(
        session_id, agent_name, prompts[agent_name]
    )
    actor = ray.get(actor_ref)  # 액터 핸들 가져오기
    
    # 원격 호출
    result_ref = actor.process_task.remote(operation, messages)
    result = ray.get(result_ref)
    return result

if __name__ == "__main__":
    Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_ray_per_session")
    ray.init(ignore_reinit_error=True)  # 데모용 로컬 클러스터; 분산을 위해 구성

    # 프롬프트 정의 (원본과 동일)
    inventory_prompt = (
        "당신은 재고 및 창고 관리 전문가입니다.\n"
        "관리할 때:\n"
        "  1) 재고/창고 과제를 분석합니다\n"
        "  2) 적절한 도구를 호출합니다\n"
        "  3) send_logistics_response로 후속 조치합니다\n"
        "비용, 효율성 및 확장성을 고려하세요."
    )
    transportation_prompt = (
        "당신은 운송 및 물류 전문가입니다.\n"
        "관리할 때:\n"
        "  1) 배송/전달 과제를 분석합니다\n"
        "  2) 적절한 도구를 호출합니다\n"
        "  3) send_logistics_response로 후속 조치합니다\n"
        "효율성, 지속가능성 및 위험 완화를 고려하세요."
    )
    supplier_prompt = (
        "당신은 공급업체 관계 및 규정 준수 전문가입니다.\n"
        "관리할 때:\n"
        "  1) 공급업체/규정 준수 문제를 분석합니다\n"
        "  2) 적절한 도구를 호출합니다\n"
        "  3) send_logistics_response로 후속 조치합니다\n"
        "성과, 규정 및 관계를 고려하세요."
    )

    prompts = {
        "inventory": inventory_prompt,
        "transportation": transportation_prompt,
        "supplier": supplier_prompt
    }

    # 세션 관리자 생성
    manager = SessionManager.remote()

    # 예시 호출
    example = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    convo = [HumanMessage(content="SKU-12345 재고가 심각하게 부족합니다. 현재 재고는 50개이지만 미처리 주문이 200개입니다. 재주문 전략은 무엇입니까?")]

    result = supervisor_invoke(example, convo, manager, prompts)
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")

    # 선택 사항: 세션별 액터 상태 쿼리
    state_ref = manager.get_session_state.remote("OP-12345", "inventory")
    if state_ref:
        state = ray.get(ray.get(state_ref))  # 중첩된 future 해결
        print("세션 액터 상태:", state)

    # Ray 종료
    ray.shutdown()
