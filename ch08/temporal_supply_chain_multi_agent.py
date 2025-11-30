from __future__ import annotations
"""
supply_chain_logistics_agent_temporal.py
내구성 있는 오케스트레이션을 위해 Temporal을 사용하는 멀티 에이전트 공급망 및 물류 관리 시스템을 위한 LangGraph 워크플로우.
Temporal 워크플로우를 통해 오케스트레이션되는 전문 에이전트를 통해 재고 관리, 운송 작업, 공급업체 관계 및 창고 최적화를 처리합니다.
워크플로우는 재시도, 영구 상태 및 장애 복구를 통해 에이전트 단계를 순차적으로 실행하며, 장기 실행 공급망 프로세스에 이상적입니다.
"""

import os
import json
from datetime import timedelta
from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any

from temporalio import workflow, activity
from temporalio.common import RetryPolicy

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.tools import tool
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions, SandboxMatcher

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

def ensure_message(m):
    if isinstance(m, BaseMessage):
        return m
    if isinstance(m, dict):
        msg_type = m.get("type")
        content = m.get("content")
        # 생성자에 전달할 kwargs에서 type 제거
        kwargs = {k:v for k,v in m.items() if k not in ["type", "content"]}
        
        if msg_type == "human":
            return HumanMessage(content=content, **kwargs)
        elif msg_type == "ai":
            return AIMessage(content=content, **kwargs)
        elif msg_type == "system":
            return SystemMessage(content=content, **kwargs)
        elif msg_type == "tool":
            return ToolMessage(content=content, **kwargs)
        return HumanMessage(content=content, **kwargs)
    return HumanMessage(content=str(m))

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
    messages: Annotated[Sequence[BaseMessage], "add"]

# Temporal 액티비티 (전문가 로직 래핑)
@activity.defn
async def supervisor_activity(operation: Dict[str, Any], messages: list) -> Dict[str, Any]:
    """슈퍼바이저를 통해 전문가를 결정하는 액티비티."""
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

    full = [SystemMessage(content=supervisor_prompt)] + [ensure_message(m) for m in messages]
    response = llm.invoke(full)
    agent_name = response.content.strip().lower()
    return {"agent_name": agent_name, "messages": [response.dict()]}

@activity.defn
async def specialist_activity(agent_name: str, operation: Dict[str, Any], messages: list) -> Dict[str, Any]:
    """전문가 처리를 위한 액티비티 (inventory, transportation, supplier)."""
    if agent_name not in prompts:
        raise ValueError(f"알 수 없는 에이전트: {agent_name}")
    
    specialist_llm = llms_dict[agent_name]
    tools = {t.name: t for t in tools_dict[agent_name]}
    system_prompt = prompts[agent_name]
    
    operation_json = json.dumps(operation, ensure_ascii=False)
    full_prompt = system_prompt + f"\n\n작업: {operation_json}"
    
    full = [SystemMessage(content=full_prompt)] + [ensure_message(m) for m in messages]

    first = specialist_llm.invoke(full)
    result_messages = [first.dict()]

    if hasattr(first, "tool_calls"):
        for tc in first.tool_calls:
            fn = tools.get(tc['name'])
            if fn:
                out = fn.invoke(tc["args"])
                result_messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]).dict())

        second = specialist_llm.invoke(full + [ensure_message(msg) for msg in result_messages])
        result_messages.append(second.dict())

    return {"messages": result_messages}

# Temporal 워크플로우
@workflow.defn(name="SupplyChainWorkflow")
class SupplyChainWorkflow:
    @workflow.run
    async def run(self, operation: Dict[str, Any], initial_messages: list) -> Dict[str, Any]:
        # 단계 1: 슈퍼바이저가 라우팅
        supervisor_result = await workflow.execute_activity(
            supervisor_activity,
            args=[operation, initial_messages],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        agent_name = supervisor_result["agent_name"]
        updated_messages = initial_messages + supervisor_result["messages"]
        
        # 단계 2: 전문가 처리
        specialist_result = await workflow.execute_activity(
            specialist_activity,
            args=[agent_name, operation, updated_messages],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # 결과 컴파일 (필요시 다단계로 확장)
        final_messages = updated_messages + specialist_result["messages"]
        return {
            "agent_name": agent_name,
            "final_messages": final_messages,
            "operation": operation
        }

# 프롬프트 정의
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

llms_dict = {
    "inventory": inventory_llm,
    "transportation": transportation_llm,
    "supplier": supplier_llm
}

tools_dict = {
    "inventory": INVENTORY_TOOLS,
    "transportation": TRANSPORTATION_TOOLS,
    "supplier": SUPPLIER_TOOLS
}

async def main():
    client = await Client.connect("localhost:7233")
    
    # 샌드박스 설정: LangChain 등 외부 라이브러리 허용
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions(
            invalid_modules=SandboxMatcher(),
            invalid_module_members=SandboxRestrictions.invalid_module_members_default,
            passthrough_modules=SandboxRestrictions.passthrough_modules_default | {
                "langchain", 
                "langchain_core", 
                "langchain_community", 
                "langchain_openai",
                "pydantic", 
                "requests", 
                "urllib3", 
                "http",
                "certifi",
                "charset_normalizer",
                "idna",
                "ssl",
                "socket",
                "logging",
                "tenacity",
                "traceloop"
            }
        )
    )

    # 워커 시작
    async with Worker(
        client, 
        task_queue="supply-chain-queue", 
        workflows=[SupplyChainWorkflow], 
        activities=[supervisor_activity, specialist_activity],
        workflow_runner=runner
    ):
        # 예시 실행
        example_operation = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
        example_messages = [{"content": "SKU-12345 재고가 심각하게 부족합니다. 현재 재고는 50개이지만 미처리 주문이 200개입니다. 재주문 전략은 무엇입니까?", "type": "human"}]

        result = await client.execute_workflow(
            SupplyChainWorkflow.run,
            args=[example_operation, example_messages],
            id="supply-chain-workflow",
            task_queue="supply-chain-queue"
        )
        print("워크플로우 결과:")
        for m in result["final_messages"]:
            print(m)

if __name__ == "__main__":
    import asyncio
    Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_temporal")
    asyncio.run(main())
