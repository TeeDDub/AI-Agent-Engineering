import asyncio
from typing import Any, Sequence, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    messages: Sequence[Any]


# MCP 도구 클라이언트 생성
mcp_client = MultiServerMCPClient(
    {
        "math": {
            "command": "python3",
            "args": ["ch04/mcp_servers/MCP_math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # `python ch04/mcp/MCP_weather_server.py`으로 MCP 서버를 먼저 실행합니다.
            "url": "http://0.0.0.0:8000/mcp",
            "transport": "streamable_http",
        },
    }
)

# MCP 도구 클라이언트에서 도구 목록 가져오기
async def get_mcp_tools() -> list[Tool]:
    return await mcp_client.get_tools()


async def call_mcp_tools(state: AgentState) -> dict[str, Any]:
    """
    state["messages"]를 포함한 AgentState를 받아 어떤 MCP 도구를 호출할지 결정합니다.
    데모 목적으로, 사용자 메시지에 'weather'나 '날씨'가 포함되어 있으면 날씨 도구를 호출하고,
    수학 표현식(숫자/연산자)이 포함되어 있으면 수학 도구를 호출합니다.
    그 외의 경우에는 기본 메시지를 반환합니다.
    """
    messages = state["messages"]
    last_msg = messages[-1].content.lower()

    # MCP_TOOLS를 전역 변수로 선언하여 한 번만 가져옵니다.
    global MCP_TOOLS
    if "MCP_TOOLS" not in globals():
        MCP_TOOLS = await mcp_client.get_tools()

    # 간단한 판단: 숫자/연산자 토큰이 있으면 "math"를, 'weather'나 '날씨'가 포함되어 있으면 "weather"를 선택합니다.
    # 실제로는 LLM이 적절한 도구를 판단하지만 여기서는 시연을 위해 간단하게 판단합니다.
    if any(token in last_msg for token in ["+", "-", "*", "/", "(", ")"]):
        tool_name = "math"
        tool_input = {"expression": messages[-1].content}
    elif "weather" in last_msg or "날씨" in messages[-1].content:
        tool_name = "weather"
        content = messages[-1].content

        if "weather in" in content.lower():
            location = content.lower().split("weather in")[1].strip().rstrip("?").strip()
        elif "의 날씨" in content:
            location = content.split("의 날씨")[0].strip()
        else:
            location = "NYC"
        tool_input = {"location": location}
    else:
        return {
            "messages": [
                {"role": "assistant", "content": "수학 또는 날씨 질문만 답변할 수 있습니다."}
            ]
        }

    tool_obj = next((t for t in MCP_TOOLS if t.name == tool_name), None)
    if tool_obj is None:
        return {
            "messages": [
                {"role": "assistant", "content": f"{tool_name} 도구를 사용할 수 없습니다."}
            ]
        }

    mcp_result: str = await tool_obj.ainvoke(tool_input)

    return {"messages": [{"role": "assistant", "content": mcp_result}]}


def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_mcp_tools)
    g.set_entry_point("assistant")
    return g.compile()


GRAPH = construct_graph()


async def run_math_query():
    initial_state = {"messages": [HumanMessage(content="(3 + 5) * 12은 얼마인가요?")]}
    result = await GRAPH.ainvoke(initial_state)
    assistant_msg = result["messages"][-1]
    content = (
        assistant_msg.get("content")
        if isinstance(assistant_msg, dict)
        else assistant_msg.content
    )
    print("Math answer:", content)


async def run_weather_query():
    initial_state = {"messages": [HumanMessage(content="NYC의 날씨는 어때요?")]}
    result = await GRAPH.ainvoke(initial_state)
    assistant_msg = result["messages"][-1]
    print("Weather answer:", assistant_msg["content"])


if __name__ == "__main__":
    asyncio.run(run_math_query())
    asyncio.run(run_weather_query())
