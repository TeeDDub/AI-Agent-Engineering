from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

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

# 도구 정의
@tool
def multiply(x: float, y: float) -> float:
    """'x'와 'y'를 곱합니다."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """'x'를 'y'제곱합니다."""
    return x**y

@tool
def add(x: float, y: float) -> float:
    """'x'와 'y'를 더합니다."""
    return x + y

tools = [multiply, exponentiate, add]

# LLM 초기화 및 도구 바인딩
llm = init_chat_model(model="gpt-5-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

query = "393 * 12.25는 얼마인가요? 그리고 11 + 49는요?"
messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {
        "add": add,
        "multiply": multiply,
        "exponentiate": exponentiate
    }[tool_call["name"]]
    result = selected_tool.invoke(tool_call['args'])
    
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"Result: {result}")
    print()
    
    # ToolMessage 생성
    tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
    messages.append(tool_msg)

final_response = llm_with_tools.invoke(messages)
print(final_response.content)