import dspy
dspy.configure(lm=dspy.LM("gpt-5-mini"))

# 위협 분류 작업을 위한 DSPy 시그니처 정의
class ThreatClassifier(dspy.Signature):
   """주어진 인디케이터(IP, URL, 해시 등)의 위협 수준을 
   'benign', 'suspicious', 'malicious' 중 하나로 분류합니다."""
   indicator: str = dspy.InputField(desc="IP 주소, URL, 파일 해시 등 분류할 인디케이터.")
   threat_level: str = dspy.OutputField(desc="분류된 위협 수준: 'benign', 'suspicious', 또는 'malicious'.")

# 사려 있는 분류를 위한 ChainOfThought 기반 DSPy 모듈
class ThreatClassificationModule(dspy.Module):
   def __init__(self):
       super().__init__()
       self.classify = dspy.ChainOfThought(ThreatClassifier)
  
   def forward(self, indicator):
       return self.classify(indicator=indicator)

# 최적화를 위한 합성/수기 주석 데이터셋(실무에서는 실제 SOC 로그에서 50~200+ 예시 사용)
# 각 예시는 인디케이터와 정답 위협 수준을 포함합니다.
trainset = [
   dspy.Example(indicator="203.0.113.45", 
       threat_level="suspicious").with_inputs('indicator'),  # 알려진 악성 IP
   dspy.Example(indicator="example.com/malware.exe", 
       threat_level="malicious").with_inputs('indicator'),  # 악성 URL
   dspy.Example(indicator="benign-site.net", 
       threat_level="benign").with_inputs('indicator'),  # 안전한 도메인
   dspy.Example(indicator="abc123def456", 
       threat_level="malicious").with_inputs('indicator'),  # 멀웨어 해시
   dspy.Example(indicator="192.168.1.1", 
       threat_level="benign").with_inputs('indicator'),  # 로컬 IP
   dspy.Example(indicator="obfuscated.url/with?params", 
       threat_level="suspicious").with_inputs('indicator'),  # 난독화 URL 엣지 케이스
   dspy.Example(indicator="new-attack-vector-hash789", 
       threat_level="malicious").with_inputs('indicator'),  # 새로운 위협
]

# 평가 지표(위협 수준의 exact match.
# 프로덕션에서는 시맨틱 매치나 커스텀 스코어러 권장)
def threat_match_metric(example, pred, trace=None):
   return example.threat_level.lower() == pred.threat_level.lower()

# 모듈 최적화(다양한 사례 처리를 위한 내부 프롬프트를 정제)
optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=threat_match_metric, 
    max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module = optimizer.compile(ThreatClassificationModule(), 
                                     trainset=trainset)

# 도구에서의 사용 예: 최적화 후 classify_threat에 사용
def classify_threat(indicator: str) -> str:
   """최적화된 DSPy 모듈을 사용해 위협 수준을 분류합니다."""
   prediction = optimized_module(indicator=indicator)
   return prediction.threat_level
