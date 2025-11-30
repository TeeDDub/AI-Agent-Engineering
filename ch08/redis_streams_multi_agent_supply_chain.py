from __future__ import annotations
"""
supply_chain_logistics_agent_redis.py
Redis Streams를 사용한 멀티 에이전트 공급망 및 물류 관리 시스템을 위한 LangGraph 워크플로우.
Redis Streams를 통해 비동기적으로 조율되는 전문 에이전트들을 통해 재고 관리, 운송 작업, 공급업체 관계, 창고 최적화를 처리합니다.
슈퍼바이저는 작업을 공유 스트림에 게시하고, 전문가들은 관련 작업을 소비 및 처리한 후 응답을 다른 스트림에 게시합니다.
"""

import os
import json
import time
import uuid
import operator
import threading
from typing import Annotated, Sequence, TypedDict, Optional

import redis

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

# 레디스
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
TASK_STREAM = 'supply-chain-tasks'
RESPONSE_STREAM = 'supply-chain-responses'

# 모든 전문가가 공유하는 도구
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

Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_redis")
llm = init_chat_model(model="gpt-5-mini", callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

# 전문 LLM에 도구 바인딩
inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm.bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # 공급망 운영 정보
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Redis용 메시지 직렬화 헬퍼
def serialize_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    return [m.model_dump() for m in messages]

# Redis에서 메시지 역직렬화 헬퍼
def deserialize_messages(serialized: list[dict]) -> Sequence[BaseMessage]:
    return [HumanMessage(**m) if m['type'] == 'human' else AIMessage(**m) if m['type'] == 'ai' else ToolMessage(**m) if m['type'] == 'tool' else SystemMessage(**m) for m in serialized]

# 슈퍼바이저: 전문가를 결정하고 Redis 스트림에 작업을 게시
def supervisor_publish(operation: dict, messages: Sequence[BaseMessage]) -> str:
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    supervisor_prompt = (
        "당신은 공급망 전문가 팀을 조율하는 슈퍼바이저입니다.\n"
        "팀 구성원:\n"
        "- inventory: 재고 수준, 예측, 품질, 창고 최적화, 확장 및 비용을 처리합니다.\n"
        "- transportation: 배송 추적, 준비, 운영 조정, 특수 처리, 반품, 배달 최적화 및 중단을 처리합니다.\n"
        "- supplier: 공급업체 평가 및 규정 준수를 처리합니다.\n"
        "\n"
        "사용자 쿼리에 따라 처리할 팀 구성원 한 명을 선택하세요.\n"
        "선택된 구성원의 이름(inventory, transportation 또는 supplier)만 출력하고, 다른 것은 출력하지 마세요.\n\n"
        f"작업: {operation_json}"
    )

    full = [SystemMessage(content=supervisor_prompt)] + messages
    response = llm.invoke(full)
    
    agent_name = response.content.strip().lower()
    
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    task_id = str(uuid.uuid4())
    task_message = {
        'task_id': task_id,
        'agent': agent_name,
        'operation': operation,
        'messages': serialize_messages(messages)
    }
    r.xadd(TASK_STREAM, {'data': json.dumps(task_message)})
    
    return task_id

# 전문가 노드 템플릿
def specialist_node(state: AgentState, specialist_llm, system_prompt: str):
    history = state["messages"]
    operation = state.get("operation", {})
    if not operation:
        operation = {"operation_id": "알 수 없음", "type": "일반", "priority": "중간", "status": "활성"}
    operation_json = json.dumps(operation, ensure_ascii=False)
    full_prompt = system_prompt + f"\n\n작업: {operation_json}"
    
    full = [SystemMessage(content=full_prompt)] + history

    first: ToolMessage | BaseMessage = specialist_llm.invoke(full)
    messages = [first]

    # 도구 호출 루프 - 최대 3번만 반복
    max_iterations = 3
    iteration = 0
    current_messages = messages

    while getattr(current_messages[-1], "tool_calls", None) and iteration < max_iterations:
        iteration += 1
        tool_messages = []
        
        for tc in current_messages[-1].tool_calls:
            print(f"[반복 {iteration}] {tc['name']}")
            # 도구 찾기 (모든 도구가 이름으로 고유하다고 가정)
            all_tools = INVENTORY_TOOLS + TRANSPORTATION_TOOLS + SUPPLIER_TOOLS
            fn = next(t for t in all_tools if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            tool_messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

        messages.extend(tool_messages)
        
        # 다음 LLM 호출
        next_message = specialist_llm.invoke(full + messages)
        messages.append(next_message)
        current_messages = [next_message]

    return {"messages": messages}

# 재고 전문가 소비자 루프
def inventory_consumer():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    last_id = '0'
    print(f"[재고 전문가] 시작됨, Redis 연결: {r.ping()}")
    inventory_prompt = (
        "당신은 재고 및 창고 관리 전문가입니다.\n"
        "관리할 때:\n"
        "  1) 재고/창고 문제를 분석합니다\n"
        "  2) 필요한 도구를 호출합니다 (예: manage_inventory, optimize_costs 등)\n"
        "  3) 작업 완료 후 send_logistics_response를 호출하여 최종 응답을 전달합니다\n"
        "  4) send_logistics_response 호출 후에는 더 이상 도구를 호출하지 않습니다\n"
        "비용, 효율성 및 확장성을 고려하세요.\n"
        "최종 응답에서는 구체적인 권장 사항과 실행 계획을 제공하세요."
    )
    
    while True:
        messages = r.xread({TASK_STREAM: last_id}, count=1, block=5000)
        if messages:
            stream, entries = messages[0]
            for entry_id, entry_data in entries:
                task = json.loads(entry_data[b'data'])
                print(f"[재고 전문가] 작업 수신: task_id={task['task_id']}, agent={task['agent']}")
                if task['agent'] == 'inventory':
                    print(f"[재고 전문가] 작업 처리 시작: {task['task_id']}")
                    state = {
                        'operation': task['operation'],
                        'messages': deserialize_messages(task['messages'])
                    }
                    result = specialist_node(state, inventory_llm, inventory_prompt)
                    response_message = {
                        'task_id': task['task_id'],
                        'from': 'inventory',
                        'result': {
                            'messages': serialize_messages(result['messages'])
                        }
                    }
                    r.xadd(RESPONSE_STREAM, {'data': json.dumps(response_message)})
                    print(f"[재고 전문가] 응답 게시됨: {task['task_id']}")
                last_id = entry_id

# 운송 전문가 소비자 루프
def transportation_consumer():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    last_id = '0'
    transportation_prompt = (
        "당신은 운송 및 물류 전문가입니다.\n"
        "관리할 때:\n"
        "  1) 배송/배달 문제를 분석합니다\n"
        "  2) 필요한 도구를 호출합니다 (예: track_shipments, arrange_shipping 등)\n"
        "  3) 작업 완료 후 send_logistics_response를 호출하여 최종 응답을 전달합니다\n"
        "  4) send_logistics_response 호출 후에는 더 이상 도구를 호출하지 않습니다\n"
        "효율성, 지속 가능성 및 위험 완화를 고려하세요.\n"
        "최종 응답에서는 구체적인 권장 사항과 실행 계획을 제공하세요."
    )
    
    while True:
        messages = r.xread({TASK_STREAM: last_id}, count=1, block=5000)
        if messages:
            stream, entries = messages[0]
            for entry_id, entry_data in entries:
                task = json.loads(entry_data[b'data'])
                if task['agent'] == 'transportation':
                    state = {
                        'operation': task['operation'],
                        'messages': deserialize_messages(task['messages'])
                    }
                    result = specialist_node(state, transportation_llm, transportation_prompt)
                    response_message = {
                        'task_id': task['task_id'],
                        'from': 'transportation',
                        'result': {
                            'messages': serialize_messages(result['messages'])
                        }
                    }
                    r.xadd(RESPONSE_STREAM, {'data': json.dumps(response_message)})
                last_id = entry_id

# 공급업체 전문가 소비자 루프
def supplier_consumer():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    last_id = '0'
    supplier_prompt = (
        "당신은 공급업체 관계 및 규정 준수 전문가입니다.\n"
        "관리할 때:\n"
        "  1) 공급업체/규정 준수 문제를 분석합니다\n"
        "  2) 필요한 도구를 호출합니다 (예: evaluate_suppliers, handle_compliance 등)\n"
        "  3) 작업 완료 후 send_logistics_response를 호출하여 최종 응답을 전달합니다\n"
        "  4) send_logistics_response 호출 후에는 더 이상 도구를 호출하지 않습니다\n"
        "성과, 규정 및 관계를 고려하세요.\n"
        "최종 응답에서는 구체적인 권장 사항과 실행 계획을 제공하세요."
    )
    
    while True:
        messages = r.xread({TASK_STREAM: last_id}, count=1, block=5000)
        if messages:
            stream, entries = messages[0]
            for entry_id, entry_data in entries:
                task = json.loads(entry_data[b'data'])
                if task['agent'] == 'supplier':
                    state = {
                        'operation': task['operation'],
                        'messages': deserialize_messages(task['messages'])
                    }
                    result = specialist_node(state, supplier_llm, supplier_prompt)
                    response_message = {
                        'task_id': task['task_id'],
                        'from': 'supplier',
                        'result': {
                            'messages': serialize_messages(result['messages'])
                        }
                    }
                    r.xadd(RESPONSE_STREAM, {'data': json.dumps(response_message)})
                last_id = entry_id

# task_id로 응답 대기 함수
def wait_for_response(task_id: str, timeout: int = 60) -> dict:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    last_id = '0'
    start_time = time.time()
    
    print(f"[대기 중] task_id={task_id}에 대한 응답을 기다립니다...")
    
    while time.time() - start_time < timeout:
        # count를 늘려서 더 많은 메시지를 한 번에 읽습니다
        messages = r.xread({RESPONSE_STREAM: last_id}, count=10, block=2000)
        if messages:
            stream, entries = messages[0]
            for entry_id, entry_data in entries:
                response = json.loads(entry_data[b'data'])
                print(f"[확인] 응답 수신: task_id={response['task_id']}, from={response['from']}")
                if response['task_id'] == task_id:
                    print(f"[성공] 일치하는 응답을 찾았습니다!")
                    return response
                # 다음 읽기를 위해 last_id 업데이트
                last_id = entry_id
    
    print(f"[타임아웃] {timeout}초 내에 응답을 받지 못했습니다")
    raise TimeoutError("제한 시간 내에 응답을 받지 못했습니다")

if __name__ == "__main__":
    # threading을 사용하여 전문가 소비자 시작
    inventory_thread = threading.Thread(target=inventory_consumer, daemon=True)
    transportation_thread = threading.Thread(target=transportation_consumer, daemon=True)
    supplier_thread = threading.Thread(target=supplier_consumer, daemon=True)

    inventory_thread.start()
    transportation_thread.start()
    supplier_thread.start()

    # 스레드가 시작될 시간을 줍니다
    time.sleep(1)

    # 예시 호출
    example_operation = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    example_messages = [HumanMessage(content="SKU-12345 재고가 심각하게 부족합니다. 현재 재고는 50개이지만 200개의 백오더가 있습니다. 재주문 전략은 무엇입니까?")]

    try:
        task_id = supervisor_publish(example_operation, example_messages)
        print(f"작업 ID로 게시됨: {task_id}")

        response = wait_for_response(task_id, timeout=90)
        print("\n=== 응답 수신 ===")
        for m in deserialize_messages(response['result']['messages']):
            print(f"\n[{m.type}]")
            print(m.content)

    except KeyboardInterrupt:
        print("\n프로그램 종료 중...")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
