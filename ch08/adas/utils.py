"""
ADAS 프레임워크를 위한 유틸리티 함수들
"""
import random
import string
import numpy as np
from scipy import stats


def random_id(length=8):
    """무작위 ID 생성"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def bootstrap_confidence_interval(data, confidence=0.95, n_bootstrap=1000):
    """부트스트랩 방법을 사용한 신뢰구간 계산"""
    if not data:
        return "0.00 ± 0.00"
    
    data = np.array(data)
    mean = np.mean(data)
    
    if len(data) < 2:
        return f"{mean:.2f} ± 0.00"
    
    # 부트스트랩 샘플링
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # 신뢰구간 계산
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, (alpha/2) * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    # 표준오차 계산
    std_error = (upper - lower) / (2 * 1.96)  # 95% CI 기준
    
    return f"{mean:.4f} ± {std_error:.4f}"


def format_multichoice_question(data):
    """MMLU 객관식 질문 포맷팅"""
    question = data.get('Question', '')
    choices = []
    
    for choice_key in ['A', 'B', 'C', 'D']:
        if choice_key in data:
            choices.append(f"{choice_key}) {data[choice_key]}")
    
    formatted = f"질문: {question}\n\n"
    formatted += "선택지:\n" + "\n".join(choices)
    
    return formatted


def format_arc_data(arc_data):
    """ARC 데이터 포맷팅"""
    task_str = "ARC 작업:\n\n"
    
    # 예제 데이터
    examples = arc_data.get('train', [])
    task_str += f"예제 {len(examples)}개:\n"
    
    for idx, example in enumerate(examples):
        task_str += f"\n예제 {idx + 1}:\n"
        task_str += f"입력: {example.get('input', [])}\n"
        task_str += f"출력: {example.get('output', [])}\n"
    
    # 테스트 입력
    test_input = None
    if 'test' in arc_data and arc_data['test']:
        test_input = arc_data['test'][0].get('input', [])
        task_str += f"\n테스트 입력: {test_input}\n"
    
    return task_str, examples, test_input


def list_to_string(lst):
    """리스트를 문자열로 변환"""
    return str(lst)


def eval_solution(prediction, arc_data, soft_eval=False):
    """ARC 솔루션 평가"""
    if prediction is None:
        return 0
    
    ground_truth = arc_data['test'][0]['output']
    
    try:
        if soft_eval:
            # 부분 점수 허용
            if prediction == ground_truth:
                return 1
            # 크기가 같은지 확인
            if len(prediction) == len(ground_truth) and len(prediction) > 0:
                if len(prediction[0]) == len(ground_truth[0]):
                    # 일치하는 셀의 비율 계산
                    total_cells = len(prediction) * len(prediction[0])
                    correct_cells = sum(
                        1 for i in range(len(prediction))
                        for j in range(len(prediction[0]))
                        if prediction[i][j] == ground_truth[i][j]
                    )
                    return correct_cells / total_cells
            return 0
        else:
            # 정확히 일치해야 함
            return 1 if prediction == ground_truth else 0
    except Exception:
        return 0

