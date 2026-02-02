#!/usr/bin/env python3
"""RLVR(Reinforcement Learning with Verifiable Rewards)로 파인튜닝한 모델 테스트 스크립트.

reinforcement_learning_with_verifiable_rewards.py로 학습한 Qwen2-0.5B 모델을 테스트합니다.
IT 헬프데스크 도구 호출 형식(<tool_call>)을 생성합니다.

사용법:
    python ch07/test_rlvr_model.py
    python ch07/test_rlvr_model.py --prompt "비밀번호를 잊어버렸습니다"
    python ch07/test_rlvr_model.py --model-path ch07/fine_tuned_model/qwen-helpdesk-rlvr/checkpoint-XXX

참고: GRPOTrainer는 save_strategy='epoch'으로 체크포인트를 저장합니다.
      output_dir에 config.json이 없으면 checkpoint-* 하위 디렉터리를 자동 탐색합니다.
"""
import argparse
import glob
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
MODEL_PATH = "ch07/fine_tuned_model/qwen-helpdesk-rlvr"

# RLVR 학습 데이터와 유사한 IT 헬프데스크 프롬프트
DEFAULT_PROMPTS = [
    "어젯밤 데이터베이스 백업이 실패했고 다음 예약된 작업은 12시간 후입니다. 긴급 복구 계획이 필요합니다.",
    "VPN 인증서가 만료되었다는 이메일을 받았습니다. 내부 네트워크에 연결할 수 없습니다.",
    "재무 부서의 신입 사원이 올바른 Active Directory 그룹에 있는데도 공유 경비 보고서 폴더에 접근할 수 없습니다.",
    "3층 로비의 프린터가 용지 걸림입니다.",
    "비밀번호를 잊어버려서 이메일이 잠겼습니다.",
]


def find_model_path(base_path: str) -> str:
    """모델 또는 최신 체크포인트 경로 반환"""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {base_path}")

    # 루트에 config.json이 있으면 전체 모델
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path

    # checkpoint-* 하위 디렉터리 탐색
    checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(
            f"{base_path}에 config.json 또는 checkpoint-* 가 없습니다. "
            "RLVR 학습을 먼저 완료했는지 확인하세요."
        )

    # 가장 큰 step 번호의 체크포인트 사용
    def step_num(p: str) -> int:
        name = os.path.basename(p)
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-")[1])
            except ValueError:
                return 0
        return 0

    latest = max(checkpoints, key=step_num)
    print(f"체크포인트 사용: {latest}")
    return latest


def load_model(model_path: str):
    """모델 및 토크나이저 로드 (RLVR는 전체 모델 저장)"""
    path = find_model_path(model_path)
    print("모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("✅ 모델 로드 완료\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """프롬프트에 대해 응답 생성 (Qwen2 chat template)"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=False)


def extract_tool_call(text: str) -> str | None:
    """생성된 텍스트에서 <tool_call> 블록 추출"""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description="RLVR 파인튜닝 모델 테스트")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="테스트할 프롬프트 (미지정 시 기본 예시들 실행)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=f"모델 또는 체크포인트 경로 (기본: {MODEL_PATH})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="생성할 최대 토큰 수",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for i, prompt in enumerate(prompts, 1):
        print(f"{'='*60}")
        print(f"[{i}] 사용자: {prompt}")
        print("-" * 60)
        response = generate(model, tokenizer, prompt, args.max_tokens)
        print(f"모델: {response.strip()}")
        tool_json = extract_tool_call(response)
        if tool_json:
            try:
                parsed = json.loads(tool_json)
                print(f"\n[추출된 도구 호출] {json.dumps(parsed, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                pass
        print()


if __name__ == "__main__":
    main()
