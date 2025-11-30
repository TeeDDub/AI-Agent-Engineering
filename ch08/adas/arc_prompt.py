"""
ARC (Abstraction and Reasoning Corpus) 작업을 위한 프롬프트 정의
"""


def get_init_archive():
    """초기 아카이브 - ARC를 위한 기본 에이전트 시스템들"""
    return [
        {
            "name": "단순 패턴 인식 에이전트",
            "code": '''
def forward(taskInfo):
    """
    단순한 패턴 인식 에이전트
    예제를 분석하고 변환 규칙을 코드로 생성합니다.
    """
    from ch08.adas.automated_design_of_agentic_systems import LLMAgentBase, Info
    
    agent = LLMAgentBase(
        output_fields=['answer'],
        agent_name='패턴분석자',
        role='ARC 퍼즐을 푸는 전문가',
        model='gpt-4o-mini',
        temperature=0.3
    )
    
    instruction = """
위 예제들을 분석하여 입력에서 출력으로의 변환 패턴을 찾으세요.
테스트 입력에 대한 출력을 list[list[int]] 형식으로만 반환하세요.
"""
    output_desc = {'answer': 'list[list[int]] 형식의 출력 그리드'}
    
    result = agent.query([taskInfo], instruction, output_desc)
    
    return result[0] if result else Info('answer', 'Agent', '[[0]]', -1)
'''
        },
        {
            "name": "코드 생성 에이전트",
            "code": '''
def forward(taskInfo):
    """
    변환 함수를 생성하는 에이전트
    예제를 분석하여 transform 함수를 작성합니다.
    """
    from ch08.adas.automated_design_of_agentic_systems import LLMAgentBase, Info
    
    # 1단계: 패턴 분석
    analyzer = LLMAgentBase(
        output_fields=['pattern_analysis'],
        agent_name='패턴분석자',
        role='시각적 패턴을 분석하는 전문가',
        model='gpt-4o-mini',
        temperature=0.5
    )
    
    analysis_instruction = "예제들의 입력-출력 패턴을 자세히 분석하세요. 색상, 위치, 크기, 대칭성 등을 고려하세요."
    analysis_desc = {'pattern_analysis': '패턴 분석 결과'}
    
    analysis = analyzer.query([taskInfo], analysis_instruction, analysis_desc)
    
    # 2단계: 코드 생성
    coder = LLMAgentBase(
        output_fields=['code'],
        agent_name='코더',
        role='Python 변환 함수를 작성하는 전문가',
        model='gpt-4o-mini',
        temperature=0.3
    )
    
    code_instruction = """
위 패턴 분석을 바탕으로 transform(input_grid) 함수를 작성하세요.
함수는 list[list[int]]를 받아서 list[list[int]]를 반환해야 합니다.
"""
    code_desc = {'code': 'transform 함수의 Python 코드'}
    
    code_result = coder.query([taskInfo] + analysis, code_instruction, code_desc)
    
    # 3단계: 코드 실행하여 답변 생성
    if code_result:
        try:
            # 코드 실행은 외부에서 처리됨
            # 여기서는 코드를 답변으로 반환
            return Info('answer', 'Agent', '[[0]]', -1)
        except Exception:
            return Info('answer', 'Agent', '[[0]]', -1)
    
    return Info('answer', 'Agent', '[[0]]', -1)
'''
        }
    ]


def get_prompt(archive):
    """새로운 에이전트 시스템을 생성하기 위한 프롬프트"""
    
    # 아카이브에서 상위 성능의 솔루션들 추출
    sorted_archive = sorted(archive, key=lambda x: x.get('fitness', '0'), reverse=True)
    
    # 시스템 프롬프트
    system_prompt = """당신은 AI 에이전트 시스템을 설계하는 전문가입니다.
ARC(Abstraction and Reasoning Corpus) 퍼즐을 풀기 위한 혁신적인 에이전트 시스템을 설계하세요.

ARC는 시각적 추론 능력을 테스트하는 퍼즐로, 입력 그리드를 분석하여 변환 규칙을 찾고 적용해야 합니다.

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

고려사항:
- 다중 에이전트 협력 방식
- 패턴 인식 알고리즘
- 코드 생성 및 검증
- 오류 처리 및 디버깅

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
1. 패턴 인식의 정확성과 포괄성
2. 에이전트 간의 정보 흐름
3. 코드 생성의 견고성
4. 엣지 케이스 처리

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
4. list[list[int]] 형식을 올바르게 처리하는가?
5. 다양한 ARC 퍼즐 유형에 대응할 수 있는가?

최종 버전을 같은 JSON 형식으로 제시하세요:
{
    "thought": "최종 설계 아이디어",
    "name": "에이전트 시스템 이름",
    "code": "최종 forward 함수 코드"
}
"""
    
    return prompt1, prompt2

