import os
import requests

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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
def query_wolfram_alpha(expression: str) -> str: 
    """
    Wolfram Alpha에 질의를 보내 식을 계산하거나 정보를 조회합니다.
    Args: expression (str): 계산하거나 평가할 수식 또는 질의입니다.
    Returns: str: 계산 결과 또는 조회된 정보입니다.
        """

    api_url = f'''https://api.wolframalpha.com/v1/result?i={requests.utils.quote(expression)}&appid={os.getenv("WOLFRAM_ALPHA_APP_ID")}'''  

    try:
        response = requests.get(api_url)
        if response.status_code == 200: 
            return response.text 
        else: raise ValueError(f'''Wolfram Alpha API 오류: 
            {response.status_code} - {response.text}''') 
    except requests.exceptions.RequestException as e: 
        raise ValueError(f"Wolfram Alpha 질의에 실패했습니다: {e}")


@tool 
def trigger_zapier_webhook(zap_id: str, payload: dict) -> str: 
    """ 미리 정의된 Zap을 실행하기 위해 Zapier 웹훅을 트리거합니다.
    Args: 
    zap_id (str): 트리거할 Zap의 고유 식별자입니다.
    payload (dict): Zapier 웹훅으로 전송할 데이터입니다.
    Returns: 
    str: Zap이 성공적으로 트리거되었을 때의 확인 메시지입니다.
    Raises: ValueError: API 요청이 실패하거나 오류를 반환한 경우 발생합니다.
    """ 

    zapier_webhook_url = f"https://hooks.zapier.com/hooks/catch/{zap_id}/" 
    try: 
        response = requests.post(zapier_webhook_url, json=payload) 
        if response.status_code == 200: 
            return f"Zapier 웹훅 '{zap_id}'이(가) 성공적으로 트리거되었습니다." 

        else: 
            raise ValueError(f'''Zapier API 오류: {response.status_code} - 
                         {response.text}''') 
    except requests.exceptions.RequestException as e: 
        raise ValueError(f"Zapier 웹훅 '{zap_id}' 트리거에 실패했습니다: {e}")

@tool 
def send_slack_message(channel: str, message: str) -> str: 
    """ 지정한 Slack 채널에 메시지를 보냅니다.
    Args: 
    channel (str): 메시지를 보낼 Slack 채널 ID 또는 이름입니다.
    message (str): 전송할 메시지의 내용입니다.
    Returns: 
    str: Slack 메시지가 성공적으로 전송되었을 때의 확인 메시지입니다.
    Raises: ValueError: API 요청이 실패하거나 오류를 반환한 경우 발생합니다.
    """ 

    api_url = "https://slack.com/api/chat.postMessage" 
    headers = { "Authorization": "Bearer YOUR_SLACK_BOT_TOKEN",
                    "Content-Type": "application/json" } 
    payload = { "channel": channel, "text": message } 
    try: 
        response = requests.post(api_url, headers=headers, json=payload) 
        response_data = response.json() 
        if response.status_code == 200 and response_data.get("ok"): 
            return f"Slack 채널 '{channel}'에 메시지가 성공적으로 전송되었습니다." 
        else: 
            error_msg = response_data.get("error", "Unknown error") 
            raise ValueError(f"Slack API 오류: {error_msg}") 
    except requests.exceptions.RequestException as e: 
        raise ValueError(f'''Slack 채널 "{channel}"로 메시지 전송에 실패했습니다: {e}''')

# LLM 초기화
llm = init_chat_model(model="gpt-4o-mini", temperature=0)
tools_list = [send_slack_message, query_wolfram_alpha, trigger_zapier_webhook]
tools_by_name = {t.name: t for t in tools_list}
llm_with_tools = llm.bind_tools(tools_list)

messages = [HumanMessage("3.15 * 12.25는 얼마인가요?")]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    chosen_tool = tools_by_name[tool_call["name"]]
    result = chosen_tool.invoke(tool_call["args"])
    tool_msg = ToolMessage(
        content=result if isinstance(result, str) else str(result),
        tool_call_id=tool_call["id"],
    )
    print(chosen_tool.name)
    print(tool_call["args"])
    print(tool_msg.content)
    messages.append(tool_msg)
    print()

final_response = llm_with_tools.invoke(messages)
print(final_response.content)
