import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 현재 파일 기준 경로 설정
base_dir = Path(__file__).resolve().parent
file_path = base_dir / "samsung_FNG_predicted.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# irrelevant_score 컬럼 존재 확인
if "irrelevant_score" not in df.columns:
    raise ValueError("irrelevant_score 컬럼이 존재하지 않습니다.")

# 0~1 범위를 100 구간으로 나누기
bin_edges = [i / 100 for i in range(101)]
bin_labels = [f"{i/100:.2f}" for i in range(100)]
df["bin"] = pd.cut(df["irrelevant_score"], bins=bin_edges, labels=bin_labels, include_lowest=True)

# 각 구간별 개수 계산
bin_counts = df["bin"].value_counts().sort_index()

# 시각화
plt.figure(figsize=(18, 6))
plt.bar(bin_counts.index, bin_counts.values)
plt.xticks(rotation=90)
plt.xlabel("irrelevant_score")
plt.ylabel("댓글 수")
plt.title("irrelevant_score raito (0~1, 0.01 unit)")
plt.tight_layout()

# 저장
save_path = base_dir / "irrelvant_raito.png"
plt.savefig(save_path)
print(f"✅ 저장 완료: {save_path}")
