import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
import re
from pathlib import Path
import argparse
import time

# ================================
# argparse: 종목명 인자 처리
# ================================
parser = argparse.ArgumentParser(description="KcBERT 감정 분류 모델 학습")
parser.add_argument("--stock_name", type=str, default="samsung", help="종목명 (예: apple, samsung 등)")
args = parser.parse_args()
stock_name = args.stock_name.lower()

# ================================
# 경로 설정
# ================================
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "../data"
model_dir = base_dir / "../model"
model_dir.mkdir(parents=True, exist_ok=True)

preprocessed_path = data_dir / f"_1_preprocessed/{stock_name}_preprocessed.csv"
labeled_path = data_dir / f"_3_labeled/{stock_name}_label.csv"

# ================================
# 1. 데이터셋 정의
# ================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
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
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# ================================
# 2. 모델 정의
# ================================
class KcBERTSentiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("beomi/KcELECTRA-base")
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = outputs[:, 0]
        logits = self.linear(self.dropout(pooled))
        probs = self.softmax(logits)
        return probs

# ================================
# 3. Threshold 스케줄 함수
# ================================
def get_threshold(epoch):
    return [1.0, 0.9, 0.7, 0.6, 0.5][min(epoch, 4)]

# ================================
# 4. 학습 루프
# ================================
def train(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", ncols=100, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ================================
# 5. 가장 최근 모델 찾기
# ================================
def find_latest_model(model_dir):
    pattern = re.compile(r"kcbert_epoch(\d+)\.pt")
    latest_epoch = -1
    latest_path = None
    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = model_dir / fname
    return latest_path, latest_epoch

# ================================
# 6. 전체 프로세스 실행
# ================================
def main():
    # 데이터 로드
    labeled_df = pd.read_csv(labeled_path)
    full_df = pd.read_csv(preprocessed_path)

    if 'content' not in labeled_df.columns or 'content' not in full_df.columns:
        raise ValueError("데이터에 'content' 컬럼이 없습니다.")

    labeled_df = labeled_df.dropna(subset=["content"])
    full_df = full_df.dropna(subset=["content"])

    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드 또는 초기화
    model = KcBERTSentiment()
    latest_model_path, latest_epoch = find_latest_model(model_dir)

    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
        print(f"✅ 가장 최근 모델 로드: {latest_model_path} (epoch {latest_epoch})")
    else:
        latest_epoch = 0
        print("🚀 새 모델로 학습 시작합니다.")

    model = model.to(device)

    # 초기 라벨 데이터셋
    labeled_texts = labeled_df["content"].tolist()
    labeled_labels = labeled_df[["fear_score", "neutral_score", "greed_score"]].values.tolist()
    labeled_dataset = SentimentDataset(labeled_texts, labeled_labels, tokenizer)
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(latest_epoch, 10):
        print(f"\n🌀 Epoch {epoch+1}")
        loss = train(model, labeled_loader, optimizer, device)
        print(f"Train loss: {loss:.4f}")

        # 모델 저장
        save_path = model_dir / f"kcbert_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"📦 저장 완료: {save_path}")

        # ---------- Pseudo-labeling ----------
        threshold = get_threshold(epoch)
        print(f"🔍 threshold = {threshold}")
        model.eval()

        full_texts = full_df["content"].tolist()
        full_dataset = SentimentDataset(full_texts, tokenizer=tokenizer)
        full_loader = DataLoader(full_dataset, batch_size=64)

        pseudo_texts, pseudo_labels = [], []
        start = time.time()
        with torch.no_grad():
            for batch in tqdm(full_loader, desc=" Inference", ncols=100):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
                confidences, _ = torch.max(outputs, dim=1)
                mask = confidences > threshold
                for i in range(len(mask)):
                    if mask[i]:
                        pseudo_texts.append(tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True))
                        pseudo_labels.append(outputs[i].cpu().numpy())
        end = time.time()
        print(f"\n✅ 전체 추론 완료 (총 시간: {end - start:.2f}초)")

        if pseudo_labels:
            print(f"  ✅ pseudo-labels accepted: {len(pseudo_labels)}")
            labeled_texts += pseudo_texts
            labeled_labels += pseudo_labels
            labeled_dataset = SentimentDataset(labeled_texts, labeled_labels, tokenizer)
            labeled_loader = DataLoader(labeled_dataset, batch_size=256, shuffle=True)
        else:
            print("  ⚠️ no pseudo-labels accepted")

if __name__ == "__main__":
    main()
