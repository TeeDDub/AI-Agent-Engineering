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
def get_pokemon_type(pokemon: str) -> str:
    """포켓몬의 타입을 가져옵니다."""
    api_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon.lower()}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            types = [t["type"]["name"] for t in data["types"]]
            return ", ".join(types)
        else:
            return f"포켓몬의 타입을 가져오는데 실패했습니다: {pokemon}"
    except requests.exceptions.RequestException:
        return f"포켓몬의 타입을 가져오는데 실패했습니다: {pokemon}"


# LLM 초기화 및 도구 바인딩
llm = init_chat_model(model="gpt-5-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_pokemon_type])

messages = [HumanMessage("피카츄의 타입은 무엇인가요?")]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_msg = get_pokemon_type.invoke(tool_call)
    
    print(tool_msg.name)
    print(tool_call['args'])
    print(tool_msg.content)
    messages.append(tool_msg)
    print()

final_response = llm_with_tools.invoke(messages)
print(final_response.content)