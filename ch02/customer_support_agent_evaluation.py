from langchain_core.messages import HumanMessage, ToolMessage
from simple_customer_support_agent import graph

# 최소 평가
example_order = {"order_id": "B73973"}
convo = [HumanMessage(content=''' 더 저렴한 곳을 찾았습니다. 
    주문 #B73973을 취소해 주세요.''')]
result = graph.invoke({"order": example_order, "messages": convo})

# 도구 호출 확인: tool_calls 속성 또는 ToolMessage 타입 확인
has_tool_call = any(
    getattr(m, "tool_calls", None) or isinstance(m, ToolMessage) 
    for m in result["messages"]
)
assert has_tool_call, "주문 취소 도구가 호출되지 않음"

# 취소 확인 메시지 확인
assert any("취소" in str(m.content) for m in result["messages"]), "확인 메시지가 누락됨"
print("✅ 에이전트가 최소 평가 기준을 통과했습니다.")
