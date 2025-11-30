from __future__ import annotations
"""
supply_chain_logistics_actor_critic.py
Actor-Critic 패턴을 적용한 멀티 에이전트 공급망 및 물류 관리 시스템.
Actor가 여러 후보 계획을 생성하고, Critic이 평가하여 최적의 계획을 선택하거나 재생성을 요청합니다.
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

# 모든 도구 리스트
ALL_TOOLS = INVENTORY_TOOLS + TRANSPORTATION_TOOLS + SUPPLIER_TOOLS

llm = init_chat_model(model="gpt-5-mini", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

# AgentState 정의 - candidates와 iteration 필드 추가
class AgentState(TypedDict):
    operation: Optional[dict]  # 공급망 운영 정보
    messages: Annotated[Sequence[BaseMessage], operator.add]
    candidates: Optional[list]  # Actor가 생성한 후보 계획들
    iteration: Optional[int]  # 반복 횟수

# Actor 노드: 후보 계획 생성
def actor_node(state: AgentState):
    """3개의 후보 공급망 계획을 생성합니다."""
    history = state["messages"]
    actor_prompt = '''3개의 공급망 후보 계획을 JSON 리스트 형식으로 생성하세요.
    형식: [{'plan': '계획 설명', 'tools': [{'tool': '도구명', 'args': {...}}]}]
    각 계획은 실행 가능한 구체적인 단계와 필요한 도구를 포함해야 합니다.'''
    response = llm.invoke([SystemMessage(content=actor_prompt)] + history)
    try:
        candidates = json.loads(response.content)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 기본 후보 제공
        candidates = [{"plan": "기본 계획", "tools": []}]
    return {"candidates": candidates, "messages": state["messages"]}

# Critic 노드: 평가 및 선택/반복
def critic_node(state: AgentState):
    """후보 계획들을 평가하고 최적의 계획을 선택하거나 재생성을 요청합니다."""
    candidates = state.get("candidates", [])
    history = state["messages"]
    
    critic_prompt = f'''다음 후보 계획들을 평가하세요: {candidates}
    
    실행 가능성(feasibility), 비용(cost), 위험도(risk) 기준으로 각각 1-10점으로 채점하세요.
    
    응답 형식 (JSON):
    {{
        "evaluations": [
            {{"plan_index": 0, "feasibility": 점수, "cost": 점수, "risk": 점수, "total": 총점}},
            ...
        ],
        "best_score": 최고점수,
        "selected": 선택된_계획_객체,
        "feedback": "개선을 위한 피드백 (점수가 8점 이하인 경우)"
    }}
    
    최고 점수가 8점 이상이면 해당 계획을 선택하고, 그렇지 않으면 재생성을 요청하세요.'''
    
    response = llm.invoke([SystemMessage(content=critic_prompt)] + history)
    
    try:
        evaluation = json.loads(response.content)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 첫 번째 후보 선택
        evaluation = {
            "best_score": 9,
            "selected": candidates[0] if candidates else {"plan": "기본 계획", "tools": []},
            "feedback": ""
        }
    
    if evaluation.get('best_score', 0) > 8:
        winning_plan = evaluation['selected']
        # 선택된 계획의 도구들을 실행
        messages = []
        for tool_info in winning_plan.get('tools', []):
            tool_name = tool_info.get('tool', '')
            tool_args = tool_info.get('args', {})
            tc = {'name': tool_name, 'args': tool_args, 'id': f'tool_{len(messages)}'}
            
            # 도구 찾기 및 실행
            try:
                fn = next(t for t in ALL_TOOLS if t.name == tool_name)
                out = fn.invoke(tool_args)
                messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
            except StopIteration:
                print(f"[경고] 도구를 찾을 수 없음: {tool_name}")
            except Exception as e:
                print(f"[오류] 도구 실행 실패: {tool_name}, {e}")
        
        # 최종 응답 전송
        send_logistics_response.invoke({"message": winning_plan.get('plan', '계획 실행 완료')})
        
        final_message = AIMessage(
            content=f"선택된 계획: {winning_plan.get('plan', '')} (점수: {evaluation.get('best_score', 0)})"
        )
        return {"messages": history + messages + [final_message]}
    else:
        # 반복: Actor에게 피드백 제공
        feedback_message = AIMessage(
            content=f"재생성 필요: 개선 사항 - {evaluation.get('feedback', '더 나은 계획이 필요합니다.')}"
        )
        return {"messages": history + [feedback_message]}

# Actor-Critic 그래프 구성
def construct_actor_critic_graph():
    """Actor-Critic 패턴을 사용한 공급망 관리 그래프를 구성합니다."""
    g = StateGraph(AgentState)
    g.add_node("actor", actor_node)
    g.add_node("critic", critic_node)
    
    g.set_entry_point("actor")
    g.add_edge("actor", "critic")
    
    # 승인되지 않은 경우 다시 Actor로 돌아감 (조건부)
    def should_continue(state: AgentState) -> str:
        """Critic이 재생성을 요청했는지 확인합니다."""
        if not state.get("messages"):
            return END
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content') and "재생성" in last_message.content:
            return "actor"
        return END
    
    g.add_conditional_edges("critic", should_continue)
    
    return g.compile()

# 그래프 컴파일
graph = construct_actor_critic_graph()

if __name__ == "__main__":
    Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_langgraph")
    print("=" * 80)
    print("공급망 물류 Actor-Critic 시스템 시작")
    print("=" * 80)
    
    # 예제 작업
    example_operation = {
        "operation_id": "OP-12345",
        "type": "inventory_management",
        "priority": "high",
        "location": "Warehouse A",
        "issue": "critical_shortage"
    }
    
    initial_message = HumanMessage(
        content="SKU-12345 재고가 심각하게 부족합니다. 현재 재고는 50개이지만 미처리 주문이 200개입니다. "
                "재주문 전략과 단기 해결책을 제시해주세요."
    )
    
    # 그래프 실행
    result = graph.invoke({
        "operation": example_operation,
        "messages": [initial_message],
        "candidates": None,
        "iteration": 0
    })
    
    print("\n" + "=" * 80)
    print("최종 결과")
    print("=" * 80)
    
    for i, m in enumerate(result["messages"], 1):
        msg_type = m.type if hasattr(m, 'type') else type(m).__name__
        content = m.content if hasattr(m, 'content') else str(m)
        print(f"\n[{i}] {msg_type}:")
        print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")
    
    print("\n" + "=" * 80)
    print("실행 완료")
    print("=" * 80)

