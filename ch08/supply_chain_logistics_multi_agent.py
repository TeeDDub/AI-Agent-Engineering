from __future__ import annotations
"""
supply_chain_logistics_multi_agent.py
멀티 에이전트 공급망 및 물류 관리 시스템을 위한 LangGraph 워크플로우.
슈퍼바이저가 조정하는 전문 에이전트를 통해 재고 관리, 운송 작업, 공급업체 관계 및 창고 최적화를 처리합니다.
"""
import os
import json
import operator
from typing import Annotated, Sequence, TypedDict, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from traceloop.sdk import Traceloop
from src.common.observability.loki_logger import log_to_loki

# 환경변수
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

llm = init_chat_model(model="gpt-5-mini", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

# 전문화된 LLM에 도구 바인딩
inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm.bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # 공급망 운영 정보
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 슈퍼바이저(관리자) 노드: 적절한 전문가에게 라우팅
def supervisor_node(state: AgentState):
    history = state["messages"]
    operation = state.get("operation", {})
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    supervisor_prompt = f"""
        당신은 공급망 전문가 팀을 조정하는 슈퍼바이저입니다.
        팀원:
        - inventory: 재고 수준, 예측, 품질, 창고 최적화, 확장 및 비용을 처리합니다.
        - transportation: 배송 추적, 준비, 운영 조정, 특수 처리, 반품, 배송 최적화 및 중단을 처리합니다.
        - supplier: 공급업체 평가 및 규정 준수를 처리합니다.

        사용자 쿼리를 기반으로 처리할 팀원 한 명을 선택하세요.
        선택한 팀원의 이름(inventory, transportation 또는 supplier)만 출력하고 다른 것은 출력하지 마세요.

        작업: {operation_json}
    """

    full = [SystemMessage(content=supervisor_prompt)] + history
    response = llm.invoke(full)
    return {"messages": [response]}

# 전문가 노드 템플릿
def specialist_node(state: AgentState, specialist_llm, system_prompt: str):
    history = state["messages"]
    operation = state.get("operation", {})
    if not operation:
        operation = {"operation_id": "UNKNOWN", "type": "general", "priority": "medium", "status": "active"}
    operation_json = json.dumps(operation, ensure_ascii=False)
    full_prompt = system_prompt + f"\n\n작업: {operation_json}"
    
    full = [SystemMessage(content=full_prompt)] + history

    first: ToolMessage | BaseMessage = specialist_llm.invoke(full)
    messages = [first]

    if getattr(first, "tool_calls", None):
        for tc in first.tool_calls:
            print(first)
            print(tc['name'])
            # 도구 찾기 (모든 도구 이름이 고유하다고 가정)
            all_tools = INVENTORY_TOOLS + TRANSPORTATION_TOOLS + SUPPLIER_TOOLS
            fn = next(t for t in all_tools if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

        second = specialist_llm.invoke(full + messages)
        messages.append(second)

    return {"messages": messages}

# 재고 전문가 노드
def inventory_node(state: AgentState):
    inventory_prompt = """
        당신은 재고 및 창고 관리 전문가입니다.
        관리할 때:
          1) 재고/창고 과제를 분석합니다
          2) 적절한 도구를 호출합니다
          3) send_logistics_response로 후속 조치합니다
        비용, 효율성 및 확장성을 고려하세요.
    """
    return specialist_node(state, inventory_llm, inventory_prompt)

# 운송 전문가 노드
def transportation_node(state: AgentState):
    transportation_prompt = """
        당신은 운송 및 물류 전문가입니다.
        관리할 때:
          1) 배송/전달 과제를 분석합니다
          2) 적절한 도구를 호출합니다
          3) send_logistics_response로 후속 조치합니다
        효율성, 지속가능성 및 위험 완화를 고려하세요.
    """
    return specialist_node(state, transportation_llm, transportation_prompt)

# 공급업체 전문가 노드
def supplier_node(state: AgentState):
    supplier_prompt = """
        당신은 공급업체 관계 및 규정 준수 전문가입니다.
        관리할 때:
          1) 공급업체/규정 준수 문제를 분석합니다
          2) 적절한 도구를 호출합니다
          3) send_logistics_response로 후속 조치합니다
        성과, 규정 및 관계를 고려하세요.
    """
    return specialist_node(state, supplier_llm, supplier_prompt)

# 조건부 엣지를 위한 라우팅 함수
def route_to_specialist(state: AgentState):
    last_message = state["messages"][-1]
    agent_name = last_message.content.strip().lower()
    if agent_name == "inventory":
        return "inventory"
    elif agent_name == "transportation":
        return "transportation"
    elif agent_name == "supplier":
        return "supplier"
    else:
        # 일치하는 항목이 없으면 폴백
        return END

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("inventory", inventory_node)
    g.add_node("transportation", transportation_node)
    g.add_node("supplier", supplier_node)
    
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_to_specialist, {"inventory": "inventory", "transportation": "transportation", "supplier": "supplier"})
    
    g.add_edge("inventory", END)
    g.add_edge("transportation", END)
    g.add_edge("supplier", END)
    
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_langgraph")
    example = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    convo = [HumanMessage(content="SKU-12345 재고가 심각하게 부족합니다. 현재 재고는 50개이지만 미처리 주문이 200개입니다. 재주문 전략은 무엇입니까?")]
    result = graph.invoke({"operation": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")
