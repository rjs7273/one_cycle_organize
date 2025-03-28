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
sample_train.to_csv("train_3class_test.csv", index=False)
sample_valid.to_csv("valid_3class_test.csv", index=False)

# ✅ 3. 모델 및 토크나이저 로딩
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# ✅ 4. 데이터셋 로드
dataset = load_dataset("csv", data_files={
    "train": "train_3class_test.csv",
    "validation": "valid_3class_test.csv"
})

# ✅ 5. 토크나이징 함수 정의
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# ✅ 6. 토큰화 적용
tokenized_dataset = dataset.map(tokenize_function)

# ✅ 7. 학습 설정 (저장 기능 포함)
training_args = TrainingArguments(
    output_dir="../0_model/kcbert_3class_test_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # ⬅️ 변경됨: 매 에폭마다 저장
    save_total_limit=1,      # ⬅️ 가장 최근 모델 하나만 저장
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="./logs_test",
    logging_steps=20
)

# ✅ 8. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # type: ignore
    eval_dataset=tokenized_dataset["validation"], # type: ignore
    tokenizer=tokenizer # type: ignore
)

# ✅ 9. 학습 시작
trainer.train()

# ✅ 10. 최종 모델 명시적으로 저장
trainer.save_model("./kcbert_3class_test_model")
tokenizer.save_pretrained("./kcbert_3class_test_model")
print("✅ 모델 저장 완료: ./kcbert_3class_test_model")
