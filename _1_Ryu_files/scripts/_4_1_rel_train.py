import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch import amp
from tqdm import tqdm
import numpy as np
import os
import re
from pathlib import Path
import argparse
import copy

# ================================
# argparse
# ================================
parser = argparse.ArgumentParser(description="KcBERT 감정 분류 모델 학습")
parser.add_argument("--stock_name", type=str, default="samsung", help="종목명")
args = parser.parse_args()
stock_name = args.stock_name.lower()

# ================================
# 하이퍼파라미터
# ================================
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 1024
PSEUDO_SAVE_BATCH = 1024

# ================================
# 경로 설정
# ================================
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "../data"
model_dir = base_dir / "../rel_model"
model_dir.mkdir(parents=True, exist_ok=True)

preprocessed_path = data_dir / f"_1_preprocessed/{stock_name}_preprocessed.csv"
labeled_path = data_dir / f"_3_labeled/{stock_name}_label.csv"
pseudo_dir = model_dir / f"pseudo_batches_{stock_name}"
pseudo_dir.mkdir(parents=True, exist_ok=True)

# ================================
# 데이터셋 정의
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
        raw_text = self.texts[idx]
        encoding = self.tokenizer(
            raw_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
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
# Threshold 스케줄
# ================================
def get_threshold(epoch):
    return [1.0, 0.9, 0.7, 0.6, 0.5][min(epoch, 4)]

# ================================
# 학습 함수
# ================================
scaler = amp.GradScaler()

def train(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", ncols=100, leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].squeeze(1).to(device)

        optimizer.zero_grad()

        with amp.autocast(device_type='cuda'):
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ================================
# 추론 함수 (model deepcopy + half 적용)
# ================================
def inference(model, tokenizer, dataset, threshold, device, batch_size=INFER_BATCH_SIZE):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    confident_data = []
    model_fp16 = copy.deepcopy(model).half().eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=" Inference", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).half()
            outputs = model_fp16(input_ids, attention_mask)

            for i, score in enumerate(outputs):
                if score > threshold:
                    text = tokenizer.decode(batch["input_ids"][i].cpu().tolist(), skip_special_tokens=True)
                    confident_data.append({
                        "text": text,
                        "reletive_score": float(score.cpu().item())
                    })
    return confident_data

# ================================
# 가장 최근 모델 찾기
# ================================
def find_latest_model(model_dir):
    pattern = re.compile(rf"kcbert_epoch_{stock_name}_(\d+)\.pt")
    latest_epoch = -1
    latest_path = None
    for path in model_dir.glob(f"kcbert_epoch_{stock_name}_*.pt"):
        match = pattern.match(path.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = path
    return latest_path, latest_epoch

# ================================
# 전체 파이프라인 실행
# ================================
def main():
    labeled_df = pd.read_csv(labeled_path)
    full_df = pd.read_csv(preprocessed_path)

    labeled_df = labeled_df.dropna(subset=["Message", "reletive_score"])
    full_df = full_df.dropna(subset=["Message"])

    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 중인 디바이스: {device}")

    model = KcBERTSentiment().to(device)
    latest_model_path, latest_epoch = find_latest_model(model_dir)
    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
        model = model.to(device).float()
        print(f"✅ 가장 최근 모델 로드: {latest_model_path} (epoch {latest_epoch})")
    else:
        latest_epoch = 0
        print("🚀 새 모델로 학습 시작합니다.")

    labeled_texts = labeled_df["Message"].tolist()
    labeled_labels = labeled_df["reletive_score"].tolist()

    pseudo_files = sorted(pseudo_dir.glob("pseudo_batch_*.csv"))
    for file in pseudo_files:
        df = pd.read_csv(file)
        df = df.dropna(subset=["text", "reletive_score"])
        labeled_texts += df["text"].tolist()
        labeled_labels += df["reletive_score"].tolist()

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(latest_epoch, 10):
        print(f"\n🌀 Epoch {epoch+1}")
        labeled_dataset = SentimentDataset(labeled_texts, labeled_labels, tokenizer)
        labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        loss = train(model, labeled_loader, optimizer, device)
        print(f"Train loss: {loss:.4f}")

        save_path = model_dir / f"kcbert_epoch_{stock_name}_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"📦 저장 완료: {save_path}")

        threshold = get_threshold(epoch)
        print(f"🔍 threshold = {threshold}")
        full_dataset = SentimentDataset(full_df["Message"].tolist(), tokenizer=tokenizer)
        batch_data = inference(model, tokenizer, full_dataset, threshold, device, batch_size=INFER_BATCH_SIZE)

        while len(batch_data) >= PSEUDO_SAVE_BATCH:
            save_idx = len(list(pseudo_dir.glob("pseudo_batch_*.csv")))
            save_path = pseudo_dir / f"pseudo_batch_{save_idx}.csv"
            if not save_path.exists():
                df_batch = pd.DataFrame(batch_data[:PSEUDO_SAVE_BATCH])
                df_batch.to_csv(save_path, index=False, encoding="utf-8-sig")
                print(f"✅ 저장됨: {save_path.name}")
            batch_data = batch_data[PSEUDO_SAVE_BATCH:]

        pseudo_files = sorted(pseudo_dir.glob("pseudo_batch_*.csv"))
        labeled_texts = labeled_df["Message"].tolist()
        labeled_labels = labeled_df["reletive_score"].tolist()
        for file in pseudo_files:
            df = pd.read_csv(file)
            labeled_texts += df["text"].tolist()
            labeled_labels += df["reletive_score"].tolist()

        cleaned = [
            (text, label)
            for text, label in zip(labeled_texts, labeled_labels)
            if isinstance(text, str) and text.strip() != "" and not pd.isna(label)
        ]
        labeled_texts = [x[0] for x in cleaned]
        labeled_labels = [x[1] for x in cleaned]

if __name__ == "__main__":
    main()
