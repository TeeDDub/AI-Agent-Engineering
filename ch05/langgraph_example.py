from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LLM 초기화
llm = init_chat_model(model="gpt-5-mini")

# State 정의
class AgentState(TypedDict):
    user_message: str
    user_id: str
    issue_type: Optional[str]
    step_result: Optional[str]
    response: Optional[str]

# 1. 노드 정의
def categorize_issue(state: AgentState) -> AgentState:
    prompt = (
        f"이 지원 요청을 'billing' 또는 'technical'로 분류하세요.\n\n"
        f"메시지: {state['user_message']}"
    )
    # invoke 사용 권장
    response = llm.invoke([HumanMessage(content=prompt)])
    kind = response.content.strip().lower()
    # 'billing'이나 'technical'이 아닌 경우 기본값 처리
    if "billing" in kind:
        kind = "billing"
    elif "technical" in kind:
        kind = "technical"
    else:
        kind = "technical" # 기본값
        
    return {"issue_type": kind}

def handle_invoice(state: AgentState) -> AgentState:
    # 인보이스 세부 정보를 조회합니다...
    return {"step_result": f"Invoice details for {state['user_id']}"}

def handle_refund(state: AgentState) -> AgentState:
    # 환불 워크플로를 시작합니다...
    return {"step_result": "Refund process initiated"}

def handle_login(state: AgentState) -> AgentState:
    # 로그인 문제를 트러블슈팅합니다...
    return {"step_result": "Password reset link sent"}

def handle_performance(state: AgentState) -> AgentState:
    # 성능 지표를 확인합니다...
    return {"step_result": "Performance metrics analyzed"}

def summarize_response(state: AgentState) -> AgentState:
    # 이전 step_result를 사용자용 메시지로 통합합니다.
    details = state.get("step_result", "")
    prompt = f"다음 내용을 바탕으로 간결한 고객 응답을 작성하세요: {details}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content.strip()}

# 2. 그래프 구성
# StateGraph 초기화 시 state_schema 전달 필수
graph_builder = StateGraph(AgentState)

# 노드 추가
graph_builder.add_node("categorize_issue", categorize_issue)
graph_builder.add_node("handle_invoice", handle_invoice)
graph_builder.add_node("handle_refund", handle_refund)
graph_builder.add_node("handle_login", handle_login)
graph_builder.add_node("handle_performance", handle_performance)
graph_builder.add_node("summarize_response", summarize_response)

# Start → categorize_issue
graph_builder.add_edge(START, "categorize_issue")

# categorize_issue → billing 또는 technical
def top_router(state: AgentState):
    return "billing" if state["issue_type"] == "billing" else "technical"

# 조건부 엣지 추가: categorize_issue -> (billing 라우터)
# 여기서는 billing/technical이 바로 다음 노드가 아니라 논리적 분기이므로
# 바로 다음 단계 함수들로 라우팅
graph_builder.add_conditional_edges(
    "categorize_issue",
    top_router,
    # 라우터의 리턴값과 다음 노드 이름 매핑
    {"billing": "handle_invoice", "technical": "handle_login"}
)

# Billing 하위 분기: invoice vs. refund
# handle_invoice에서 다음 단계로 넘어갈 때 조건부 분기
def billing_router(state: AgentState):
    msg = state["user_message"].lower()
    return "refund" if "refund" in msg else "invoice_end"

# handle_invoice 실행 후 -> 환불이 필요하면 refund로, 아니면 바로 요약으로
graph_builder.add_conditional_edges(
    "handle_invoice",
    billing_router,
    {"refund": "handle_refund", "invoice_end": "summarize_response"}
)

# Technical 하위 분기: login vs. performance
# handle_login 실행 후 -> 성능 문제가 있으면 performance로, 아니면 바로 요약으로
def tech_router(state: AgentState):
    msg = state["user_message"].lower()
    return "performance" if "performance" in msg else "login_end"

graph_builder.add_conditional_edges(
    "handle_login",
    tech_router,
    {"performance": "handle_performance", "login_end": "summarize_response"}
)



# 3. 그래프 실행
if __name__ == "__main__":
    initial_state = {
        "user_message": "안녕하세요, 인보이스와 (가능하다면) 환불 관련 도움을 받고 싶습니다.",
        "user_id": "U1234"
    }
    
    # invoke 사용
    result = graph.invoke(initial_state)
    print(result["response"])
