"""
MMLU 작업을 위한 프롬프트 정의
"""


def get_init_archive():
    """초기 아카이브 - MMLU를 위한 기본 에이전트 시스템들"""
    return [
        {
            "name": "단순 추론 에이전트",
            "code": '''
def forward(taskInfo):
    """
    단순한 직접 답변 에이전트
    질문을 읽고 바로 답변을 생성합니다.
    """
    from ch08.adas.automated_design_of_agentic_systems import LLMAgentBase, Info
    
    agent = LLMAgentBase(
        output_fields=['answer'],
        agent_name='답변자',
        role='MMLU 문제를 푸는 전문가',
        model='gpt-4o-mini',
        temperature=0.3
    )
    
    instruction = "위 질문을 읽고 가장 적절한 답을 선택하세요. 알파벳(A, B, C, D) 중 하나만 반환하세요."
    output_desc = {'answer': '당신의 답변. A, B, C, D 중 하나만 반환하세요.'}
    
    result = agent.query([taskInfo], instruction, output_desc)
    
    return result[0] if result else Info('answer', 'Agent', 'A', -1)
'''
        },
        {
            "name": "사고의 연쇄 에이전트",
            "code": '''
def forward(taskInfo):
    """
    Chain-of-Thought 추론을 사용하는 에이전트
    단계별로 생각하고 답변을 도출합니다.
    """
    from ch08.adas.automated_design_of_agentic_systems import LLMAgentBase, Info
    
    # 1단계: 추론 에이전트
    reasoner = LLMAgentBase(
        output_fields=['reasoning'],
        agent_name='추론자',
        role='논리적으로 사고하는 전문가',
        model='gpt-4o-mini',
        temperature=0.5
    )
    
    reasoning_instruction = "이 문제를 단계별로 분석하고 각 선택지를 평가하세요."
    reasoning_desc = {'reasoning': '단계별 추론 과정'}
    
    reasoning_result = reasoner.query([taskInfo], reasoning_instruction, reasoning_desc)
    
    # 2단계: 답변 에이전트
    answerer = LLMAgentBase(
        output_fields=['answer'],
        agent_name='답변자',
        role='추론을 바탕으로 최종 답변을 선택하는 전문가',
        model='gpt-4o-mini',
        temperature=0.2
    )
    
    answer_instruction = "위 추론을 바탕으로 최종 답을 선택하세요. A, B, C, D 중 하나만 반환하세요."
    answer_desc = {'answer': '최종 답변. A, B, C, D 중 하나'}
    
    result = answerer.query([taskInfo] + reasoning_result, answer_instruction, answer_desc)
    
    return result[0] if result else Info('answer', 'Agent', 'A', -1)
'''
        }
    ]


def get_prompt(archive):
    """새로운 에이전트 시스템을 생성하기 위한 프롬프트"""
    
    # 아카이브에서 상위 성능의 솔루션들 추출
    sorted_archive = sorted(archive, key=lambda x: x.get('fitness', '0'), reverse=True)
    
    # 시스템 프롬프트
    system_prompt = """당신은 AI 에이전트 시스템을 설계하는 전문가입니다.
MMLU(Massive Multitask Language Understanding) 문제를 풀기 위한 혁신적인 에이전트 시스템을 설계하세요.

당신의 목표는 기존 아카이브의 솔루션들을 분석하고, 더 나은 성능을 내는 새로운 에이전트 시스템을 만드는 것입니다."""

    # 사용자 프롬프트
    prompt = "# 기존 아카이브:\n\n"
    
    for idx, solution in enumerate(sorted_archive[:3]):  # 상위 3개만 표시
        prompt += f"## 솔루션 {idx + 1}: {solution.get('name', '이름 없음')}\n"
        prompt += f"성능: {solution.get('fitness', 'N/A')}\n"
        prompt += f"```python\n{solution.get('code', '')}\n```\n\n"
    
    prompt += """
# 작업:
기존 솔루션들을 개선한 새로운 에이전트 시스템을 설계하세요.

다음 JSON 형식으로 응답하세요:
{
    "thought": "설계 아이디어와 개선 전략",
    "name": "새로운 에이전트 시스템의 이름",
    "code": "forward(taskInfo) 함수의 Python 코드"
}

forward 함수는 taskInfo를 입력받아 Info 객체를 반환해야 합니다.
"""
    
    return system_prompt, prompt


def get_reflexion_prompt(prev_solution):
    """성찰(Reflexion)을 위한 프롬프트"""
    
    prompt1 = """
방금 생성한 솔루션을 다시 검토하세요.

다음을 고려하여 개선점을 찾으세요:
1. 코드의 정확성과 효율성
2. 에이전트 간의 협력 방식
3. 프롬프트 엔지니어링의 품질
4. 잠재적인 오류나 엣지 케이스

같은 JSON 형식으로 개선된 버전을 제시하세요:
{
    "thought": "개선된 설계 아이디어",
    "reflection": "이전 버전의 문제점과 개선 사항",
    "name": "에이전트 시스템 이름",
    "code": "개선된 forward 함수 코드"
}
"""
    
    prompt2 = """
최종 검토를 수행하세요.

다음을 확인하세요:
1. 코드가 문법적으로 올바른가?
2. 모든 필요한 import가 포함되어 있는가?
3. Info 객체를 올바르게 반환하는가?
4. MMLU 문제 형식에 맞게 설계되었는가?

최종 버전을 같은 JSON 형식으로 제시하세요:
{
    "thought": "최종 설계 아이디어",
    "name": "에이전트 시스템 이름",
    "code": "최종 forward 함수 코드"
}
"""
    
    return prompt1, prompt2

