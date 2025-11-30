from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 함숫값 또는 모델 호출을 Runnable로 감쌉니다
# RunnableLambda는 callable을 직접 인자로 받습니다.
llm_model = init_chat_model(model="gpt-5-mini", temperature=0)
llm = RunnableLambda(llm_model.invoke)

prompt = RunnableLambda(lambda text: 
    PromptTemplate.from_template(text).format_prompt().to_messages())

# 기존 체인과 동일한 형태:
# chain = LLMChain(prompt=prompt, llm=llm)
# 파이프 연산자를 사용하는 LCEL 체인:
chain = prompt | llm

# 체인 실행
if __name__ == "__main__":
    result = chain.invoke("프랑스의 수도는 어디인가요?")
    print(result.content)
