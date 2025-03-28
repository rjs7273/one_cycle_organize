"""
2024년 1월 1일 ~ 2025년 3월 10일 중 임의의 날짜 5개를 추출 후
선택한 시점 기준 최근 댓글 100개를 수집해
삼성전자_testing_{i}.csv로 저장하는 코드
"""

import pandas as pd
import numpy as np
import random
import os

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

# 기준 시점보다 정확히 1일(24시간) 이전 데이터의 가중치(weight)는 약 0.3679입니다.
# one_cycle 기준으로는 1일 이후의 데이터는 의미없다고 판단할거임
# 메인 데이터로 모델링 시에는 3일 정도가 적당할듯

# 가중치 계산 함수
def compute_time_weight(df: pd.DataFrame, tau: int = 86400) -> pd.DataFrame: # 이 tau를 조정하면 됨
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce')
    df = df.dropna(subset=["timestamp"])

    latest_time = df["timestamp"].max()
    df["delta_seconds"] = (latest_time - df["timestamp"]).dt.total_seconds()
    df["weight"] = np.exp(-df["delta_seconds"] / tau)
    return df

# 입력 / 출력 경로
input_path = '../0_data/1_preprocessed/삼성전자_preprocess.csv'
output_dir = '../0_data/4_test_data'
os.makedirs(output_dir, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(input_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce')
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# 날짜 범위 설정
start_date = pd.to_datetime("2024-01-01", utc=True)
end_date = pd.to_datetime("2025-03-10", utc=True)

# 임의 날짜 5개 샘플링
random_dates = sorted(random.sample(
    list(pd.date_range(start=start_date, end=end_date, freq="D")), 5
),
reverse=True)

# 날짜별 100개씩 추출 및 저장
for i, ref_date in enumerate(random_dates, 1):
    # 기준 시점 이후의 데이터 추출
    sub_df = df[df["timestamp"] >= ref_date].sort_values("timestamp").head(100).copy()

    # 만약 100개 미만이라면 패스
    if len(sub_df) < 100:
        print(f"⚠️ {ref_date.date()} 기준 데이터가 100개 미만이므로 건너뜀")
        continue

    # 시간 가중치 계산
    sub_df = compute_time_weight(sub_df)

    # 저장
    output_path = os.path.join(output_dir, f'삼성전자_testing_{i}.csv')
    sub_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ {output_path} 저장 완료. 기준 날짜: {ref_date.date()}")

print("🎉 전체 샘플링 및 저장 완료.")
