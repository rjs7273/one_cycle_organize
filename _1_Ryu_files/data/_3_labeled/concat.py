import pandas as pd
from pathlib import Path

# 현재 파일 위치 기준
base_dir = Path(__file__).resolve().parent
backup_dir = base_dir / "backup"
output_path = base_dir / "samsung_label.csv"

# samsung_label_*.csv 파일 모두 불러오기
csv_files = sorted(backup_dir.glob("samsung_label_*.csv"))
df_list = [pd.read_csv(file) for file in csv_files]

# 병합 및 저장
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {output_path.name}")
else:
    print("❌ 병합할 파일이 없습니다.")
