import pandas as pd
import numpy as np
import os

# 종목명과 기준 날짜 설정
stock_name = "samsung"         # 예: samsung, skhynix, apple, nvidia
ref_date_str = "2025-12-15"    # 기준 날짜 (KST 기준)
ref_date = pd.to_datetime(ref_date_str).tz_localize("Asia/Seoul")

# 경로 설정
input_path = f'_0_data/_3_predict/{stock_name}_predict_bert.csv'

# 가중치 계산 함수
def compute_time_weight(df: pd.DataFrame, tau: int = 86400) -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["time"], errors='coerce')
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.tz_convert("Asia/Seoul")  # 이미 tz-aware이면 convert
    latest_time = df["time"].max()
    df["delta_seconds"] = (latest_time - df["time"]).dt.total_seconds()
    df["weight"] = np.exp(-df["delta_seconds"] / tau)
    return df

# 데이터 불러오기 및 전처리
df = pd.read_csv(input_path)

df["time"] = pd.to_datetime(df["time"], errors='coerce')
df["time"] = df["time"].dt.tz_convert("Asia/Seoul")
df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

# 기준 시점 이전 데이터에서 최근 100개 수집
sub_df = df[df["time"] < ref_date].sort_values("time", ascending=False).head(100).copy()
sub_df = sub_df.sort_values("time").reset_index(drop=True)

# 100개 미만이면 경고 출력 후 종료
if len(sub_df) < 100:
    print(f"⚠️ {ref_date.date()} 기준 {stock_name} 데이터가 100개 미만이므로 계산하지 않음")
else:
    # 가중치 계산
    sub_df = compute_time_weight(sub_df)

    # 개별 감성 점수 계산
    sub_df['individual_score'] = (
        sub_df['prob_fear'] * 0 +
        sub_df['prob_neutral'] * 50 +
        sub_df['prob_greed'] * 100
    )

    # 가중치 반영 점수 계산
    sub_df['weighted_score'] = sub_df['individual_score'] * sub_df['weight']

    # 최종 점수 계산
    total_weighted_score = sub_df['weighted_score'].sum()
    total_weight = sub_df['weight'].sum()
    final_score = total_weighted_score / total_weight if total_weight > 0 else 0

    # 출력
    print(f"📅 기준 날짜: {ref_date.date()} (KST)")
    print(f"🧪 샘플링 수: {len(sub_df)}개")
    print(f"📊 커뮤니티 감성 지수 (공포/탐욕 점수): {final_score:.2f}")

print("🎉 작업 완료.")
