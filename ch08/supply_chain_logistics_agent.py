from __future__ import annotations
"""
supply_chain_logistics_agent.py
재고 관리, 운송 작업, 공급업체 관계 및 창고 최적화를 처리하는 
공급망 및 물류 관리 에이전트를 위한 LangGraph 워크플로우.
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

# 환경변수
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"

@tool
def manage_inventory(sku: Optional[str] = None, **kwargs) -> str:
    """재고 수준, 재고 보충, 감사 및 최적화 전략을 관리합니다."""
    print(f"[도구] manage_inventory(sku={sku}, kwargs={kwargs})")
    log_to_loki("tool.manage_inventory", f"sku={sku}")
    return "inventory_management_initiated"

@tool
def track_shipments(origin: Optional[str] = None, **kwargs) -> str:
    """배송 상태, 지연 사항을 추적하고 배송 물류를 조정합니다."""
    print(f"[도구] track_shipments(origin={origin}, kwargs={kwargs})")
    log_to_loki("tool.track_shipments", f"origin={origin}")
    return "shipment_tracking_updated"

@tool
def evaluate_suppliers(supplier_name: Optional[str] = None, **kwargs) -> str:
    """공급업체 성과를 평가하고 감사를 수행하며 공급업체 관계를 관리합니다."""
    print(f"[도구] evaluate_suppliers(supplier_name={supplier_name}, kwargs={kwargs})")
    log_to_loki("tool.evaluate_suppliers", f"supplier_name={supplier_name}")
    return "supplier_evaluation_complete"

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
def handle_compliance(compliance_type: Optional[str] = None, **kwargs) -> str:
    """규제 준수, 세관, 문서화 및 인증을 관리합니다."""
    print(f"[도구] handle_compliance(compliance_type={compliance_type}, kwargs={kwargs})")
    log_to_loki("tool.handle_compliance", f"compliance_type={compliance_type}")
    return "compliance_handled"

@tool
def process_returns(returned_quantity: Optional[str] = None, **kwargs) -> str:
    """반품, 역물류 및 제품 처리를 처리합니다."""
    print(f"[도구] process_returns(returned_quantity={returned_quantity}, kwargs={kwargs})")
    log_to_loki("tool.process_returns", f"returned_quantity={returned_quantity}")
    return "returns_processed"

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

@tool
def send_logistics_response(operation_id: Optional[str] = None, message: Optional[str] = None) -> str:
    """이해관계자에게 물류 업데이트, 권장 사항 또는 상태 보고서를 전송합니다."""
    print(f"[도구] send_logistics_response → {message}")
    log_to_loki("tool.send_logistics_response", f"operation_id={operation_id}, message={message}")
    return "logistics_response_sent"

TOOLS = [
    manage_inventory, track_shipments, evaluate_suppliers, optimize_warehouse,
    forecast_demand, manage_quality, arrange_shipping, coordinate_operations,
    manage_special_handling, handle_compliance, process_returns, scale_operations,
    optimize_costs, optimize_delivery, manage_disruption, send_logistics_response
]


llm = init_chat_model(model="gpt-5-mini", callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # 공급망 운영 정보
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # 누락되거나 불완전한 작업 데이터를 적절히 처리
    operation = state.get("operation", {})
    if not operation:
        operation = {"operation_id": "UNKNOWN", "type": "general", "priority": "medium", "status": "active"}
    
    operation_json = json.dumps(operation, ensure_ascii=False)
    system_prompt = f"""
        당신은 숙련된 공급망 및 물류 관리 전문가입니다.
        전문 분야:
        - 재고 관리 및 수요 예측
        - 운송 및 배송 최적화
        - 공급업체 관계 관리 및 평가
        - 창고 운영 및 용량 계획
        - 품질 관리 및 규정 준수 관리
        - 비용 최적화 및 운영 효율성
        - 위험 관리 및 중단 대응
        - 지속가능성 및 친환경 물류 이니셔티브

        공급망 운영을 관리할 때:
        1) 물류 과제 또는 기회를 분석합니다
        2) 적절한 공급망 관리 도구를 호출합니다
        3) send_logistics_response로 권장 사항을 제공합니다
        4) 비용, 효율성, 품질 및 지속가능성 영향을 고려합니다
        5) 고객 만족도와 비즈니스 연속성을 우선시합니다

        항상 비용 최적화와 서비스 품질 및 위험 완화의 균형을 유지하십시오.

        작업: {operation_json}"""

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
    Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_langgraph")
    example = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    convo = [HumanMessage(content="SKU-12345 재고가 심각하게 부족합니다. 현재 재고는 50개이지만 미처리 주문이 200개입니다. 재주문 전략은 무엇입니까?")]
    result = graph.invoke({"operation": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}") 