import pandas as pd
import matplotlib.pyplot as plt

# ✅ 1. 예측 결과 로드
final_df = pd.read_csv("../0_data/3_predict/삼성전자_predict_bert.csv")

# ✅ 2. 시간 정보 로드 및 결합
filtered_df = pd.read_csv("../0_data/1_preprocess/삼성전자_filtered.csv")
filtered_df = filtered_df[filtered_df["text_bert"].notnull()]  # ✅ null 제거

# ✅ 3. time 열 병합
final_df["time"] = filtered_df["time"].reset_index(drop=True)

# ✅ 4. datetime 형식으로 변환
final_df["time"] = pd.to_datetime(final_df["time"], errors="coerce")

# ✅ 5. 시간 단위(Hour)로 내림
final_df["hour"] = final_df["time"].dt.floor("H")

# ✅ 6. 시간대별 평균 공포탐욕지수 계산
hourly_avg = final_df.groupby("hour")["공포탐욕지수"].mean().reset_index()
hourly_avg.to_csv("hourly_feargreed_score_bert.csv", index=False, encoding="utf-8-sig")
print("✅ 시간대별 평균 저장 완료")

# ✅ 7. 최근 1주일 데이터 필터링
latest = hourly_avg["hour"].max().normalize()
week_ago = latest - pd.Timedelta(days=6)
weekly_avg = hourly_avg[(hourly_avg["hour"] >= week_ago) & (hourly_avg["hour"] <= latest)].copy()
weekly_avg["hour"] = pd.to_datetime(weekly_avg["hour"]).dt.tz_localize(None)

# ✅ 8. 한글 폰트 설정 (맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 9. 시각화 및 저장
plt.figure(figsize=(12, 6))
plt.plot(weekly_avg["hour"], weekly_avg["공포탐욕지수"], marker='o', linestyle='-', color='blue', label='공포탐욕지수')
plt.title(f"{week_ago.date()} ~ {latest.date()} 시간대별 평균 공포탐욕지수")
plt.xlabel("시간")
plt.ylabel("공포탐욕지수 (0~100)")
plt.axhline(50, color='gray', linestyle='--', label="중립 기준선")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./chart/hourly_fear_and_greed.png")
print("✅ 그래프 저장 완료: ./chart/hourly_fear_and_greed.png")
