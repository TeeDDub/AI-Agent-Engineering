import os
import requests
import logging
import numpy as np
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 환경변수 확인
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

# LLM 초기화
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# 도구 정의
@tool 
def query_wolfram_alpha(expression: str) -> str: 
    """
    Wolfram Alpha에 질의를 보내 식을 계산하거나 정보를 조회합니다.
    Args: expression (str): 계산하거나 평가할 수식 또는 질의입니다.
    Returns: str: 계산 결과 또는 조회된 정보입니다.
        """

    api_url = f'''https://api.wolframalpha.com/v1/result?i={requests.utils.quote(expression)}&appid=YOUR_WOLFRAM_ALPHA_APP_ID'''  

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

# 도구 그룹 정의
tool_groups = {
    "Computation": {
        "description": "수학 계산 및 데이터 분석과 관련된 도구입니다.",
        "tools": [query_wolfram_alpha]
    },
    "Automation": {
        "description": "워크플로우를 자동화하고 다양한 서비스를 통합하는 도구입니다.",
        "tools": [trigger_zapier_webhook]
    },
    "Communication": {
        "description": "커뮤니케이션 및 메시징을 돕는 도구입니다.",
        "tools": [send_slack_message]
    }
}

# -------------------------------
# LLM 기반 계층적 스킬 선택
# -------------------------------
def select_group_llm(query: str) -> str:
    """
    LLM을 사용하여 사용자의 쿼리를 기반으로 가장 적절한 스킬 그룹을 결정합니다.
    
    Args:
        query (str): 사용자의 입력 쿼리.
        
    Returns:
        str: 선택된 그룹의 이름.
    """
    prompt = f"다음 쿼리에 가장 적절한 스킬 그룹을 선택하세요: '{query}'. 스킬 그룹명만 반환하세요.\n선택지는 다음과 같습니다: [Computation, Automation, Communication]."
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def select_tool_llm(query: str, group_name: str) -> str:
    """
    LLM을 사용하여 사용자의 쿼리를 기반으로 그룹 내에서 가장 적절한 도구를 결정합니다.
    
    Args:
        query (str): 사용자의 입력 쿼리.
        group_name (str): 선택된 스킬 그룹의 이름.
        
    Returns:
        str: 선택된 도구 함수의 이름.
    """
    prompt = f"쿼리: '{query}'를 기반으로, 그룹 '{group_name}'에서 가장 적절한 도구를 선택하세요."
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# 사용자 쿼리 예시
user_query = "2x + 3 = 7"

# 1단계: LLM을 사용하여 가장 관련성 높은 스킬 그룹 선택
selected_group_name = select_group_llm(user_query)
if not selected_group_name:
    print("쿼리에 적합한 스킬 그룹을 찾을 수 없습니다.")
else:
    # 그룹 이름에 마침표 등이 포함될 수 있으므로 정리 (예: "Computation." -> "Computation")
    selected_group_name = selected_group_name.replace(".", "")
    
    logging.info(f"선택된 그룹: {selected_group_name}")
    print(f"선택된 스킬 그룹: {selected_group_name}")

    if selected_group_name not in tool_groups:
        print(f"오류: 선택된 그룹 '{selected_group_name}'은(는) 유효한 그룹이 아닙니다.")
    else:
        # 2단계: LLM을 사용하여 그룹 내에서 가장 관련성 높은 도구 선택
        selected_tool_name = select_tool_llm(user_query, selected_group_name)
        
        # 도구 이름 정리 (예: "query_wolfram_alpha." -> "query_wolfram_alpha")
        selected_tool_name = selected_tool_name.replace(".", "")
        
        selected_tool = globals().get(selected_tool_name, None)
        
        if not selected_tool:
            print("선택된 그룹 내에서 적합한 도구를 찾을 수 없습니다.")
        else:
            logging.info(f"선택된 도구: {selected_tool.__name__}")
            print(f"선택된 도구: {selected_tool.__name__}")
            
            # 도구에 따른 인자 준비
            args = {}
            if selected_tool == query_wolfram_alpha:
                # 전체 쿼리를 표현식으로 가정
                args["expression"] = user_query
            elif selected_tool == trigger_zapier_webhook:
                # 데모용 placeholder 사용
                args["zap_id"] = "123456"
                args["payload"] = {"message": user_query}
            elif selected_tool == send_slack_message:
                # 데모용 placeholder 사용
                args["channel"] = "#general"
                args["message"] = user_query
            else:
                print("선택된 도구를 인식할 수 없습니다.")
            
            # 선택된 도구 호출
            try:
                tool_result = selected_tool.invoke(args)
                print(f"도구 '{selected_tool.__name__}' 결과: {tool_result}")
            except ValueError as e:
                print(f"오류: {e}")
