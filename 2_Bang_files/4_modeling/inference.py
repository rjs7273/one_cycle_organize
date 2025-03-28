import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ 1. 모델 및 토크나이저 로드
model_path = "./kcbert_3class_test_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ 2. 예측 대상 데이터 로드
df = pd.read_csv("../0_data/1_preprocess/삼성전자_filtered.csv")

# ✅ (테스트용) 샘플 1000개만 추출 → 전체 추론 시 아래 줄 주석 처리
df = df.sample(n=1000, random_state=42).reset_index(drop=True)

texts = df["text_bert"].fillna("").tolist()

# ✅ 3. 배치 추론
results = []
batch_size = 32

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)  # [batch_size, 3]
    
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
final_df.to_csv("../0_data/3_predict/삼성전자_predict_bert.csv", index=False, encoding="utf-8-sig")
print("✅ 추론 결과 저장 완료: 삼성전자_predict_bert.csv")
