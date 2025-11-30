# memory_evaluation.py

from typing import List, Tuple, Dict, Any
import statistics

def precision_recall_f1(predicted: List[Any], expected: List[Any]) -> Tuple[float, float, float]:
    """
    두 항목 리스트 간의 정밀도(precision), 재현율(recall), F1 점수를 계산합니다.
    항목들은 동등성을 기준으로 비교됩니다. 순서는 중요하지 않습니다.
    """
    pred_set = set(predicted)
    exp_set = set(expected)
    if not pred_set and not exp_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set)
    recall = tp / len(exp_set) if exp_set else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def evaluate_memory_updates(
    predicted_updates: List[Any],
    expected_updates: List[Any]
) -> Dict[str, float]:
    """
    에이전트의 메모리 업데이트가 예상 업데이트와 얼마나 잘 일치하는지 평가합니다.
    정밀도, 재현율, F1 점수를 반환합니다.
    """
    p, r, f1 = precision_recall_f1(predicted_updates, expected_updates)
    return {"memory_precision": p, "memory_recall": r, "memory_f1": f1}

def evaluate_memory_retrieval(
    retrieve_fn: Any,
    queries: List[str],
    expected_results: List[List[Any]],
    top_k: int = 1
) -> Dict[str, float]:
    """
    k개의 메모리 항목 리스트를 반환하는 검색 함수 `retrieve_fn(query, k)`가 주어졌을 때,
    여러 쿼리에 대해 평가합니다.
    반환값:
      - `retrieval_accuracy@k`: 상위 k개 결과 중 적어도 하나의 예상 항목이 포함된 쿼리의 비율입니다.
    """
    hits = 0
    for query, expect in zip(queries, expected_results):
        results = retrieve_fn(query, top_k)
        # 예상 항목을 검색했는지 확인합니다.
        if set(results) & set(expect):
            hits += 1
    accuracy = hits / len(queries) if queries else 1.0
    return {f"retrieval_accuracy@{top_k}": accuracy}

def aggregate_metrics(list_of_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    지표 딕셔너리들의 리스트(예: evaluate_*의 출력)가 주어졌을 때,
    각 지표의 평균을 계산합니다.
    """
    if not list_of_dicts:
        return {}
    aggregated: Dict[str, float] = {}
    keys = list_of_dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in list_of_dicts if k in d]
        aggregated[k] = statistics.mean(vals) if vals else 0.0
    return aggregated
