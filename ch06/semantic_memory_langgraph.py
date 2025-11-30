from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START

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

# LLM 초기화
llm = init_chat_model(model="gpt-5-mini", temperature=0)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


from vectordb import Memory

memory = Memory(chunking_strategy={'mode':'sliding_window', 'window_size': 128, 'overlap': 16})

text = """
머신러닝은 분석 모델 구축을 자동화하는 데이터 분석 방법이다.

이는 시스템이 데이터로부터 학습하고 패턴을 식별하며 최소한의 인간 개입으로 의사결정을 내릴 수 있다는 개념에 기반한 인공지능의 한 분야이다.

머신러닝 알고리즘은 원하는 출력 사례를 포함한 데이터셋으로 학습된다. 예를 들어, 이미지를 분류하는 머신러닝 알고리즘은 고양이와 개의 이미지를 포함한 데이터셋으로 훈련될 수 있다.

알고리즘이 학습을 마치면 새로운 데이터에 대한 예측에 사용될 수 있다. 예를 들어, 이미지 분류 알고리즘은 새로운 이미지에 고양이가 있는지 개인지가 있는지를 예측하는 데 활용될 수 있다.
"""

metadata = {"title": "Introduction to Machine Learning", "url": "https://learn.microsoft.com/en-us/training/modules/introduction-to-machine-learning"}

memory.save(text, metadata)

text2 = """
인공지능(AI)은 인간처럼 사고하고 행동을 모방하도록 프로그래밍된 기계에서 인간 지능을 시뮬레이션하는 것을 의미한다.

이 용어는 학습과 문제 해결과 같이 인간의 정신과 연관된 특성을 보이는 모든 기계에 적용될 수 있다.

AI 연구는 게임 플레이부터 의료 진단에 이르기까지 매우 다양한 문제를 해결하기 위한 효과적인 기법을 개발하는 데 큰 성공을 거두었다.
"""

metadata2 = {"title": "Introduction to Artificial Intelligence", "url": "https://microsoft.github.io/AI-For-Beginners/"}

memory.save(text2, metadata2)

query = "AI와 머신러닝은 어떤 관계가 있나요?"

results = memory.search(query, top_n=3)

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile()

input_message = {"type": "user", "content": "안녕하세요! 제 이름은 민혁입니다."}
for chunk in graph.stream({"messages": [input_message]}, {}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print(results)
