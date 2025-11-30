import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 예제 1: 수치형 기능 편향(drift)에 대한 Kolmogorov-Smirnov (KS) 검정 (예: 쿼리 길이)
def detect_ks_drift(historical_data: np.ndarray, current_data: np.ndarray, threshold: float = 0.1) -> bool:
    """
    KS 검정을 사용하여 분포 변화를 감지합니다.
    유의미한 편향(통계량 > 임계값)이 있으면 True를 반환합니다.
    """
    ks_stat, p_value = stats.ks_2samp(historical_data, current_data)
    print(f"KS 통계량: {ks_stat}, p-값: {p_value}")
    return ks_stat > threshold

# 사용 예시
historical_lengths = np.array([len(q) for q in ["What is the weather?", "Book a flight to Paris", "Recommend a book"] * 100])
current_lengths = np.array([len(q) for q in ["Query about latest AI news", "Longer user input with details"] * 150])
if detect_ks_drift(historical_lengths, current_lengths):
    print("편향 감지됨: 입력 변경 사항을 검토하세요.")

# 예제 2: 확률 분포 변화에 대한 Kullback-Leibler (KL) 발산 (예: 토큰 분포)
def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    두 확률 분포 간의 KL 발산을 계산합니다.
    0으로 나누는 것을 방지하기 위해 엡실론을 추가합니다.
    """
    # 두 배열의 길이를 맞춥니다 (작은 쪽을 기준으로 자르거나 0으로 채움)
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]

    p = p + epsilon
    q = q + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

# 사용 예시 (토큰 빈도 히스토그램)
historical_tokens = np.bincount([ord(c) for q in ["hello world"] * 100 for c in q], minlength=256)  # 단순화된 토큰 카운트
current_tokens = np.bincount([ord(c) for q in ["hola mundo"] * 100 for c in q], minlength=256)
kl_score = kl_divergence(historical_tokens, current_tokens)
print(f"KL 발산: {kl_score}")
if kl_score > 0.5:
    print("개념 편향 감지됨: 언어 변화 가능성.")

# 예제 3: 범주형 지표에 대한 모집단 안정성 지수 (PSI) (예: 도구 사용)
def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    범주형 또는 구간화된 연속형 데이터에 대한 PSI를 계산합니다.
    """
    expected_percents = expected / np.sum(expected)
    actual_percents = actual / np.sum(actual)
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)

# 사용 예시 (도구 호출 횟수)
historical_tools = np.array([50, 30, 20])  # 예: 'refund', 'cancel', 'modify'에 대한 카운트
current_tools = np.array([40, 40, 20])
psi = calculate_psi(historical_tools, current_tools)
print(f"PSI: {psi}")
if psi > 0.1:
    print("도구 사용에 경미한 편향 발생.")
elif psi > 0.25:
    print("주요 편향 발생: 개입 필요.")

# 예제 4: 쿼리 편향에 대한 임베딩 기반 유사도
def detect_embedding_drift(historical_queries: list, current_queries: list, threshold: float = 0.8):
    """
    쿼리 임베딩 간의 평균 코사인 유사도를 계산합니다.
    단순함을 위해 TF-IDF를 사용합니다. 더 나은 의미론적 분석을 위해 sentence transformers로 대체하세요.
    """
    vectorizer = TfidfVectorizer()
    all_queries = historical_queries + current_queries
    embeddings = vectorizer.fit_transform(all_queries)
    hist_emb = embeddings[:len(historical_queries)]
    curr_emb = embeddings[len(historical_queries):]
    similarities = cosine_similarity(curr_emb, hist_emb)
    mean_sim = np.mean(similarities)
    print(f"평균 코사인 유사도: {mean_sim}")
    return mean_sim < threshold

# 사용 예시
historical = ["Refund my order", "Cancel shipment", "Change address"] * 50
current = ["Return damaged item", "Stop delivery now", "Update shipping info"] * 50
if detect_embedding_drift(historical, current):
    print("쿼리 편향 감지됨: 프롬프트를 재학습하거나 조정하세요.")