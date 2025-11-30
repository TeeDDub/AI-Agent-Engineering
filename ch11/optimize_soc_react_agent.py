import dspy
dspy.configure(lm=dspy.LM("gpt-5-mini"))

def lookup_threat_intel(indicator: str) -> str:
    """모의: 인디케이터에 대한 위협 인텔리전스를 조회합니다."""
    return f"Mock intel for {indicator}: potentially malicious"

def query_logs(query: str) -> str:
    """모의: 보안 로그를 검색하고 분석합니다."""
    return f"Mock logs for '{query}': suspicious activity detected"

# 소수의 합성 테스트 케이스(경보 → 기대 응답)
# 실제로는 실제 로그에서 파생하거나 실패 사례에 주석을 달아 수집합니다.
# 더 나은 최적화를 위해 100개+를 목표로 하세요.
trainset = [
    dspy.Example(alert='''Suspicious login attempt from IP 203.0.113.45 to 
                 admin account.''',
                 response='''Lookup threat intel for IP, query logs for activity, 
                     triage as true positive, isolate host if malicious.''')
                     .with_inputs('alert'),
    dspy.Example(alert="Unusual file download from URL example.com/malware.exe.",
                 response='''Lookup threat intel for URL and hash, query logs 
                     for endpoint activity, triage as true positive, isolate 
                     host.''').with_inputs('alert'),
    dspy.Example(alert="High network traffic to domain suspicious-site.net.",
                 response='''Lookup threat intel for domain, query logs for 
                     network and firewall, triage as false positive if 
                     benign.''').with_inputs('alert'),
    dspy.Example(alert='''Alert: Potential phishing email with attachment 
                 hash abc123.''',
                 response='''Lookup threat intel for hash, query logs for email 
                     and endpoint, triage as true positive, send analyst 
                     response.''').with_inputs('alert'),
    dspy.Example(alert='''Anomaly in user behavior: multiple failed logins from 
                 new device.''',
                 response='''Query logs for authentication, lookup threat intel 
                     for device IP, triage as true positive if pattern matches 
                     attack.''').with_inputs('alert'),
]

# SOC 인시던트 처리를 위한 리액트 모듈 정의
react = dspy.ReAct("alert -> response", tools=[lookup_threat_intel, query_logs])

# 단순 지표를 사용하는 옵티마이저
# (예시를 위해 exact match 사용. 프로덕션에서는
# 시맨틱 유사도 같은 더 정교한 지표를 사용하세요)
tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", 
                  num_threads=24)
optimized_react = tp.compile(react, trainset=trainset)
