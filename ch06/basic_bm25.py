from rank_bm25 import BM25Okapi
from typing import List

corpus: List[List[str]] = [
    "에이전트 J는 패기가 넘치는 신입 대원이다".split(),
    "에이전트 K는 수년간의 MIB 경험과 멋진 뉴럴라이저를 갖고 있다".split(),
    "두 명의 에이전트가 검은 정장을 입고 은하계를 구했다".split(),
]
# 2. BM25 인덱스 생성
bm25 = BM25Okapi(corpus)

# 3. 간단한 쿼리로 검색 수행
query = "신입 대원은 누구지?".split()
top_n = bm25.get_top_n(query, corpus, n=2)

print("쿼리:", " ".join(query))
print("상위 일치 문장:")
for line in top_n:
    print(" •", " ".join(line))
