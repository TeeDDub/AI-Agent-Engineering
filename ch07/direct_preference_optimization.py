# fine_tune_helpdesk_dpo.py
# DPO(Direct Preference Optimization)를 사용한 헬프데스크 모델 파인튜닝
import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
import logging

BASE_SFT_CKPT = "microsoft/Phi-3-mini-4k-instruct"
DPO_DATA      = "ch07/training_data/dpo_it_help_desk_training_data.jsonl"                   # -> 경로 또는 HF 데이터셋
OUTPUT_DIR    = "ch07/fine_tuned_model/phi3-mini-helpdesk-dpo"

# 1️⃣ 모델 + 토크나이저 로드
tok = AutoTokenizer.from_pretrained(BASE_SFT_CKPT, padding_side="right",
                                    trust_remote_code=True)

logger = logging.getLogger(__name__)
if not os.path.exists(BASE_SFT_CKPT):
    logger.warning("로컬 경로를 찾을 수 없습니다. Hub에서 '%s'를 다운로드합니다.", BASE_SFT_CKPT)

# 2️⃣ 양자화 설정 (4비트)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # 가중치를 4비트로 양자화
    bnb_4bit_use_double_quant=True,  # 선택사항: 중첩 양자화
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3️⃣ 기본 모델 로드 (양자화 적용)
base = AutoModelForCausalLM.from_pretrained(
    BASE_SFT_CKPT,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

# 4️⃣ LoRA 설정 및 모델 준비
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                    "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)
print("✅  Phi-3 모델 로드 완료:", model.config.hidden_size, "hidden dim")

# 5️⃣ 데이터셋 로드
ds = load_dataset("json", data_files=DPO_DATA, split="train")

# 6️⃣ 학습 인자 설정 (사용하지 않음, DPOConfig로 대체)
train_args = TrainingArguments(
    output_dir      = OUTPUT_DIR,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate   = 5e-6,
    num_train_epochs= 3,
    logging_steps   = 10,
    save_strategy   = "epoch",
    bf16            = True,
    report_to       = None,
)

# 7️⃣ DPO 학습 설정
dpo_args = DPOConfig(
    output_dir              = "phi3-mini-helpdesk-dpo",
    per_device_train_batch_size  = 4,
    gradient_accumulation_steps  = 4,
    learning_rate           = 5e-6,
    num_train_epochs        = 3.0,
    bf16                    = True,
    logging_steps           = 10,
    save_strategy           = "epoch",
    report_to               = None,
    beta                    = 0.1,
    loss_type               = "sigmoid",
    label_smoothing         = 0.0,
    max_prompt_length       = 4096,
    max_completion_length   = 4096,
    max_length              = 8192,
    padding_value           = tok.pad_token_id,
    label_pad_token_id      = tok.pad_token_id,
    truncation_mode         = "keep_end",
    generate_during_eval    = False,
    disable_dropout         = False,
    reference_free          = True,
    model_init_kwargs       = None,
    ref_model_init_kwargs   = None,
)

# 8️⃣ DPO 트레이너 초기화
trainer = DPOTrainer(
    model,
    ref_model=None,           # reference_free=True이므로 참조 모델 불필요
    args=dpo_args,
    train_dataset=ds,
)

# 9️⃣ 학습 실행 및 모델 저장
trainer.train()
trainer.save_model()
tok.save_pretrained(OUTPUT_DIR)
print(f"✅  모델 저장 완료: {OUTPUT_DIR}")
