import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# ✅ 1. 기존 데이터에서 샘플링
train_df = pd.read_csv("../0_data/2_labeling/삼성전자_train_3class.csv")
valid_df = pd.read_csv("../0_data/2_labeling/삼성전자_valid_3class.csv")

sample_train = train_df.sample(3000, random_state=42)
sample_valid = valid_df.sample(1000, random_state=42)

# ✅ 2. 샘플 저장
sample_train.to_csv("../0_data/2_labeling/삼성전자_train_3class_sample.csv", index=False)
sample_valid.to_csv("../0_data/2_labeling/삼성전자_valid_3class_sample.csv", index=False)

# ✅ 3. 모델 및 토크나이저 로딩
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# ✅ 4. 디바이스 설정 (GPU 사용 여부 확인 후 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 모델을 GPU로 이동 (CPU면 CPU 사용)

# ✅ 5. 데이터셋 로드
dataset = load_dataset("csv", data_files={
    "train": "../0_data/2_labeling/삼성전자_train_3class_sample.csv",
    "validation": "../0_data/2_labeling/삼성전자_valid_3class_sample.csv"
})

# ✅ 6. 토크나이징 함수 정의
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# ✅ 7. 토큰화 적용
tokenized_dataset = dataset.map(tokenize_function)

# ✅ 8. 학습 설정
training_args = TrainingArguments(
    output_dir="../0_model/kcbert_3class_test_model",  # ✅ 모델 저장 위치
    evaluation_strategy="epoch",
    save_strategy="epoch",  # ✅ 에포크마다 자동 저장
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    logging_dir="./logs_test",
    logging_steps=20
)

# ✅ 9. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # type: ignore
    eval_dataset=tokenized_dataset["validation"], # type: ignore
    tokenizer=tokenizer # type: ignore
)

# ✅ 10. 학습 시작
trainer.train()

# ✅ 11. 최종 모델 수동 저장 (선택적 중복 저장)
trainer.save_model("../0_model/kcbert_3class_test_model")
tokenizer.save_pretrained("../0_model/kcbert_3class_test_model")

print("✅ 학습 및 저장 완료: ../0_model/kcbert_3class_test_model")
