from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import requests

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

@tool
def get_stock_price(ticker: str) -> float:
    """주식 시장 거래소 거래 티커에 대한 주식 가격을 가져옵니다."""
    api_url = f"https://api.example.com/stocks/{ticker}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()["price"]
        else:
            return f"주식 가격을 가져오는데 실패했습니다: {ticker}"
    except requests.exceptions.RequestException:
        return f"주식 가격을 가져오는데 실패했습니다: {ticker}"


# LLM 초기화 및 도구 바인딩
llm = init_chat_model(model="gpt-5-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_stock_price])

messages = [HumanMessage("애플의 현재 주식 가격은 얼마인가요?")]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_msg = get_stock_price.invoke(tool_call)
    
    print(tool_msg.name)
    print(tool_call['args'])
    print(tool_msg.content)
    messages.append(tool_msg)
    print()

final_response = llm_with_tools.invoke(messages)
print(final_response.content)