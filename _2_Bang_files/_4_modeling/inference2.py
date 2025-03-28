import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--stock", type=str, required=True, help="종목명 (예: 삼성전자, sk하이닉스, 애플, 엔비디아)")
args = parser.parse_args()
stock = args.stock

# ✅ 1. 모델 및 토크나이저 로드
model_path = f"./kcbert_3class_{stock}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ 2. 예측 대상 데이터 로드
data_path = f"../0_data/1_preprocess/{stock}_filtered.csv"
df = pd.read_csv(data_path)
df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
texts = df["text_bert"].fillna("").tolist()

# ✅ 3. 배치 추론
results = []
batch_size = 32

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    for prob in probs:
        prob = prob.tolist()
        label = int(torch.argmax(torch.tensor(prob)))
        results.append({
            "prob_fear": round(prob[0], 4),
            "prob_neutral": round(prob[1], 4),
            "prob_greed": round(prob[2], 4),
            "pred_label": label
        })

# ✅ 4. 결과 결합 및 저장
predict_df = pd.DataFrame(results)
final_df = pd.concat([df.reset_index(drop=True), predict_df], axis=1)
predict_path = f"../0_data/3_predict/{stock}_predict_bert.csv"
final_df.to_csv(predict_path, index=False, encoding="utf-8-sig")
print(f"✅ 추론 결과 저장 완료: {predict_path}")
