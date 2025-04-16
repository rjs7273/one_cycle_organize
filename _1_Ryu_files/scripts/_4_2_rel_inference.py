import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import argparse
import os
import re
import numpy as np
import copy

# ================================
# argparse
# ================================
parser = argparse.ArgumentParser(description="KcBERT 감정 분류 모델 추론")
parser.add_argument("--stock_name", type=str, default="samsung", help="종목명")
args = parser.parse_args()
stock_name = args.stock_name.lower()

# ================================
# 경로 설정
# ================================
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "../data"
model_dir = base_dir / "../rel_model"
model_dir.mkdir(parents=True, exist_ok=True)

preprocessed_path = data_dir / f"_1_preprocessed/{stock_name}_preprocessed.csv"
out_path = data_dir / f"_4_predicted/{stock_name}_rel_predicted.csv"

# ================================
# 하이퍼파라미터
# ================================
INFER_BATCH_SIZE = 1024

# ================================
# 데이터셋 정의
# ================================
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw_text = self.texts[idx]
        encoding = self.tokenizer(
            raw_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["raw_text"] = raw_text
        return item

# ================================
# 모델 정의
# ================================
class KcBERTSentiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("beomi/kcbert-base")
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = outputs[:, 0]
        logits = self.linear(self.dropout(pooled))
        return self.activation(logits).squeeze(-1)

# ================================
# 가장 최근 모델 찾기
# ================================
def find_latest_model(model_dir):
    pattern = re.compile(rf"kcbert_epoch_{stock_name}_(\d+)\.pt")
    latest_epoch = -1
    latest_path = None
    for path in Path(model_dir).glob(f"kcbert_epoch_{stock_name}_*.pt"):
        match = pattern.match(path.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = path
    return latest_path, latest_epoch

# ================================
# 추론 함수
# ================================
def run_inference():
    df = pd.read_csv(preprocessed_path)
    df = df.dropna(subset=["Message"])
    texts = df["Message"].tolist()

    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    dataset = SentimentDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=INFER_BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KcBERTSentiment().to(device)

    model_path, epoch = find_latest_model(model_dir)
    if model_path is None:
        raise ValueError("모델 파일이 존재하지 않습니다.")
    print(f"✅ 모델 로드: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = copy.deepcopy(model).half().eval()  # ✅ half precision for inference

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).half()
            outputs = model(input_ids, attention_mask)  # shape: (batch,)
            scores = outputs.cpu().numpy()
            for i, score in enumerate(scores):
                results.append({
                    "text": batch["raw_text"][i],
                    "reletive_score": float(score)
                })

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 추론 결과 저장 완료: {out_path}")

if __name__ == "__main__":
    run_inference()
