import pandas as pd
from pathlib import Path

# 현재 파일 기준 디렉토리
base_dir = Path(__file__).resolve().parent

# samsung_label_*.csv 파일 모두 찾기
csv_files = list(base_dir.glob("samsung_label_*.csv"))

# 모든 파일을 DataFrame으로 읽고 합치기
df_all = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# 결과 확인
print(f"총 {len(csv_files)}개 파일을 병합했습니다.")
df_all.to_csv("samsung_label_merged.csv", index=False, encoding="utf-8-sig")