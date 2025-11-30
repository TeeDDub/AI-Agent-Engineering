from typing import TypedDict, Annotated, Sequence
import operator
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph

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

# State 타입 정의
class AgentState(TypedDict):
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]

# -- 1) 주문 취소 도구 정의
@tool
def cancel_order(order_id: str) -> str:
    """배송되지 않은 주문을 취소합니다."""
    # (여기서 실제 백엔드 API를 호출합니다)
    return f"주문 {order_id}이(가) 취소되었습니다."


# -- 2) 에이전트 구조 정의: LLM 호출, 도구 실행, 다시 LLM 호출
def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})

    # LLM 초기화
    llm = init_chat_model(model="gpt-5-mini", temperature=0)
    llm_with_tools = llm.bind_tools([cancel_order]) # 도구 바인딩

    # 시스템 프롬프트는 모델이 할 일을 정확히 알려줍니다
    prompt = (
        f'''당신은 이커머스 지원 에이전트입니다.
        주문 ID: {order['order_id']}
        고객이 취소를 요청하면 cancel_order(order_id)를 호출하고
        간단한 확인 메시지를 보내세요.
        그렇지 않으면 일반적으로 응답하세요.'''
    )
    full = [SystemMessage(content=prompt)] + msgs

    # 1차 LLM 패스: 도구 호출 여부 결정
    first = llm_with_tools.invoke(full)
    out = [first]

    if getattr(first, "tool_calls", None):
        # cancel_order 도구 실행
        tc = first.tool_calls[0]
        result = cancel_order.invoke(tc["args"])
        out.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # 2차 LLM 패스: 최종 확인 텍스트 생성
        second = llm.invoke(full + out)
        out.append(second)

    return {"messages": out}

# -- 3) StateGraph로 에이전트 구조 연결
def construct_graph():
    g = StateGraph(AgentState)  # TypedDict 사용
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    example_order = {"order_id": "B73973"}
    convo = [HumanMessage(content="주문 #B73973를 취소해주세요.")]
    result = graph.invoke({"order": example_order, "messages": convo})
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")
