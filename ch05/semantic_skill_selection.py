import os
import requests
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

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


# OpenAI 임베딩 및 LLM 초기화
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 도구 설명
tool_descriptions = {
    "query_wolfram_alpha": "Wolfram Alpha에 질의를 보내 식을 계산하거나 정보를 조회합니다.",
    "trigger_zapier_webhook": "미리 정의된 Zap을 실행하기 위해 Zapier 웹훅을 트리거합니다.",
    "send_slack_message": "지정한 Slack 채널에 메시지를 보냅니다."
}

# 각 도구 설명에 대한 임베딩 생성
tool_embeddings = []
tool_names = []

for tool_name, description in tool_descriptions.items():
    embedding = embeddings.embed_query(description)
    tool_embeddings.append(embedding)
    tool_names.append(tool_name)

# FAISS 벡터 저장소 초기화
dimension = len(tool_embeddings[0])  # 모든 임베딩의 차원이 동일하다고 가정
index = faiss.IndexFlatL2(dimension)

# 코사인 유사도를 위한 임베딩 정규화
faiss.normalize_L2(np.array(tool_embeddings).astype('float32'))

# 리스트를 FAISS 호환 형식으로 변환
tool_embeddings_np = np.array(tool_embeddings).astype('float32')
index.add(tool_embeddings_np)

# 인덱스를 도구 함수에 매핑
index_to_tool = {
    0: "query_wolfram_alpha",
    1: "trigger_zapier_webhook",
    2: "send_slack_message"
}

def select_tool(query: str, top_k: int = 1) -> list:
    """
    벡터 기반 검색을 사용하여 사용자 질의에 가장 적합한 도구(들)를 선택합니다.
    
    Args:
        query (str): 사용자의 입력 질의.
        top_k (int): 검색할 상위 도구의 수.
        
    Returns:
        list: 선택된 도구 함수 이름의 리스트.
    """
    query_embedding = np.array(embeddings.embed_query(query)).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    selected_tools = [index_to_tool[idx] for idx in I[0] if idx in index_to_tool]
    return selected_tools

def determine_parameters(query: str, tool_name: str) -> dict:
    """
    LLM을 사용하여 질의를 분석하고 호출할 도구의 파라미터를 결정합니다.
    
    Args:
        query (str): 사용자의 입력 질의.
        tool_name (str): 선택된 도구 이름.
        
    Returns:
        dict: 도구에 사용할 파라미터.
    """
    messages = [
        HumanMessage(content=f"사용자의 질의: '{query}'를 바탕으로, 도구 '{tool_name}'에 사용할 파라미터는 무엇입니까?")
    ]
    
    # LLM을 호출하여 파라미터 추출
    response = llm.invoke(messages)
    
    # LLM 응답을 파싱하는 예제 로직
    parameters = {}
    
    # 주의: 실제 응답은 AIMessage 객체이며, content 속성에 텍스트가 포함됩니다.
    # 아래 로직은 예시이며 실제로는 JSON 파싱이나 구조화된 출력을 사용해야 합니다.
    # 여기서는 데모를 위해 기본값이나 간단한 매핑을 사용합니다.
    
    if tool_name == "query_wolfram_alpha":
        # 예시: 응답에서 식을 추출한다고 가정
        parameters["expression"] = query # 간단히 전체 쿼리를 사용
    elif tool_name == "trigger_zapier_webhook":
        parameters["zap_id"] = "123456"  # 기본 Zap ID
        parameters["payload"] = {"data": query}
    elif tool_name == "send_slack_message":
        parameters["channel"] = "#general"
        parameters["message"] = query
    
    return parameters

# 예제 사용자 질의
user_query = "2x + 3 = 7"

# 상위 도구 선택
selected_tools = select_tool(user_query, top_k=1)
tool_name = selected_tools[0] if selected_tools else None

if tool_name:
    # 질의와 선택된 도구를 기반으로 LLM을 사용하여 파라미터 결정
    args = determine_parameters(user_query, tool_name)

    # 선택된 도구 호출
    try:
        # 주의: 실제 도구 함수(query_wolfram_alpha 등)가 globals()에 정의되어 있어야 합니다.
        # 이 파일에는 도구 구현체가 포함되어 있지 않으므로 실행 시 에러가 발생할 수 있습니다.
        if tool_name in globals():
            tool_result = globals()[tool_name].invoke(args)
            print(f"도구 '{tool_name}' 결과: {tool_result}")
        else:
            print(f"도구 '{tool_name}'가 정의되지 않았습니다. (실제 실행을 위해서는 도구 함수 구현이 필요합니다)")
            # 디버깅용 출력
            print(f"선택된 도구: {tool_name}")
            print(f"파라미터: {args}")

    except ValueError as e:
        print(f"도구 '{tool_name}' 호출 중 오류 발생: {e}")
else:
    print("선택된 도구가 없습니다.")
