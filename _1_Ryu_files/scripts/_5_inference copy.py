import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import argparse
import re
import os
from pathlib import Path
from tqdm import tqdm

# ================================
# argparse: 종목명 인자 처리
# ================================
parser = argparse.ArgumentParser(description="KcBERT 감정 예측")
parser.add_argument("--stock_name", type=str, default="samsung", help="종목명 (예: samsung, apple 등)")
args = parser.parse_args()
stock_name = args.stock_name.lower()

# ================================
# 경로 설정
# ================================
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "../data"
model_dir = base_dir / "../model"

preprocessed_path = data_dir / f"_1_preprocessed/{stock_name}_preprocessed.csv"
output_path = data_dir / f"_4_predicted/{stock_name}_predicted.csv"

# preprocessed_path = data_dir / f"_1_preprocessed/samsung_sample_preprocessed.csv" # 샘플 데이터로 테스트
# output_path = data_dir / f"_4_predicted/samsung_sample_predicted.csv" # 샘플 데이터 출력결과

# ================================
# 데이터셋 클래스
# ================================
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

# ================================
# 모델 클래스
# ================================
class KcBERTSentiment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("beomi/kcBERT-base")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = outputs[:, 0]
        logits = self.linear(self.dropout(pooled))
        probs = self.softmax(logits)
        return probs

# ================================
# 가장 최근 모델 찾기
# ================================
def find_latest_model(model_dir):
    pattern = re.compile(rf"kcbert_epoch_{stock_name}_(\d+)\.pt")
    latest_epoch = -1
    latest_path = None
    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = model_dir / fname
    return latest_path

# ================================
# 추론 실행
# ================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcBERT-base")
    model = KcBERTSentiment()
    model = model.to(device)    


    latest_model = find_latest_model(model_dir)
    if latest_model is None:
        raise FileNotFoundError("모델이 존재하지 않습니다.")
    model.load_state_dict(torch.load(latest_model, map_location=device))
    print(f"✅ 모델 로드: {latest_model}")

    # 데이터 로드
    df = pd.read_csv(preprocessed_path).dropna(subset=["Message"])
    texts = df["Message"].tolist()
    dataset = SentimentDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=1024)

    # 추론
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="추론 중", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            all_probs.extend(outputs.cpu().tolist())

    # 결과 저장
    probs_array = torch.tensor(all_probs).numpy()
    df["fear_score"] = probs_array[:, 0]
    df["neutral_score"] = probs_array[:, 1]
    df["greed_score"] = probs_array[:, 2]
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"📁 추론 결과 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
