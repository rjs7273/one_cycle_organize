import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ✅ 현재 파일 기준 경로 설정
base_dir = Path(__file__).resolve().parent / "../data/_4_predicted"
file_path = base_dir / "samsung_FNG_predicted.csv"

# ✅ CSV 로드 및 날짜 파싱
df = pd.read_csv(file_path)
df["Updated At"] = pd.to_datetime(df["Updated At"], utc=True)

# ✅ 추세선 계산 (index 기준 rolling)
window_size = 200
# df["fear_trend"] = df["fear"].rolling(window=window_size, min_periods=1).mean()
df["greed_trend"] = df["greed"].rolling(window=window_size, min_periods=1).mean()

# ✅ 저장 경로 생성
output_dir = base_dir / "charts"
output_dir.mkdir(parents=True, exist_ok=True)

# ✅ 1만 개씩 분할 저장
chunk_size = 10000
num_chunks = (len(df) + chunk_size - 1) // chunk_size

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(df))
    chunk = df.iloc[start:end]

    # ✅ 날짜를 x축으로 지정
    x_dates = chunk["Updated At"]

    plt.figure(figsize=(15, 6))
    # plt.plot(x_dates, chunk["fear_trend"], label="Fear (Trend)", color="blue")
    plt.plot(x_dates, chunk["greed_trend"], label="Greed (Trend)", color="green")
    plt.title(f"Samsung Stock: {x_dates.iloc[0].strftime('%Y-%m-%d')} ~ {x_dates.iloc[-1].strftime('%Y-%m-%d')}")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ✅ 파일 저장
    output_path = output_dir / f"samsung_trend_{start}_{end}.png"
    plt.savefig(output_path)
    plt.close()

print(f"✅ 저장 완료: {num_chunks}개의 PNG 파일이 {output_dir}에 생성되었습니다.")
