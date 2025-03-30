import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict

# # ✅ 커맨드라인 인자 방식
# # 사용법 : python train2.py --stock samsung
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--stock", type=str, required=True)
# args = parser.parse_args()

# stock_name = args.stock

# # ✅ 변수 인자 방식
# stock_list = ["samsung","skhynix","apple","nvidia"]

stock_name = "apple"

# ✅ 경로 구성
train_path = f"_0_data/_2_labeling/{stock_name}_train_3class.csv"
valid_path = f"_0_data/_2_labeling/{stock_name}_valid_3class.csv"
output_dir = f"_0_model/kcbert_3class_{stock_name}"

# ✅ 데이터 로드 및 샘플링 (파일 저장은 생략)
train_df = pd.read_csv(train_path).sample(3000, random_state=42)
valid_df = pd.read_csv(valid_path).sample(1000, random_state=42)

# 결측치 제거
train_df = train_df.dropna(subset=["text_bert"])
valid_df = valid_df.dropna(subset=["text_bert"])

# ✅ HuggingFace Dataset 변환
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(valid_df)
})

# ✅ 모델 및 토크나이저 로딩
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# ✅ 토크나이징
def tokenize_function(example):
    return tokenizer(example["text_bert"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=20
)

# ✅ Trainer 구성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer # type: ignore
)

trainer.train()

# ✅ 모델 저장
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ 모델 저장 완료: {output_dir}")
