import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # ✅ 커맨드라인 인자 방식
# # 사용법 : python inference2.py --stock samsung
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--stock", type=str, required=True)
# args = parser.parse_args()

# stock_name = args.stock

# ✅ 변수 인자 방식
# stock_list = ["samsung","skhynix","apple","nvidia"]

stock_name = "skhynix"

# ✅ 경로 구성
model_path = f"_0_model/kcbert_3class_{stock_name}"
input_path = f"_0_data/_1_preprocess/{stock_name}_filtered.csv"
output_path = f"_0_data/_3_predict/{stock_name}_predict_bert.csv"

# ✅ 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ 데이터 로드
df = pd.read_csv(input_path)
df = df.sample(n=1000, random_state=41).reset_index(drop=True)  # 전체 추론 시 주석 처리
texts = df["text_bert"].fillna("").tolist()

# ✅ 배치 추론
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

# ✅ 저장
predict_df = pd.DataFrame(results)
final_df = pd.concat([df.reset_index(drop=True), predict_df], axis=1)
final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 추론 결과 저장 완료: {output_path}")
