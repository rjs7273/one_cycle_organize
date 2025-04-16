"""
인코딩 문제 해결 & 불필요한 컬럼 제거 & float 타입으로 변환하는 임시 코드
"""

import pandas as pd

# 파일 경로 설정
input_path = "/root/one_cycle_organize/_1_Ryu_files/data/_3_labeled/samsung_label.csv"
output_path = "/root/one_cycle_organize/_1_Ryu_files/data/_3_labeled/samsung_label.csv"

# 삭제할 컬럼 목록
drop_columns = ["reason", "old_irrelevant_score", "old_ fear_score", "old_greed_score", "new_reason"]

# CP949로 파일 읽기
df = pd.read_csv(input_path)

# 불필요한 컬럼 제거 (존재하는 경우에만)
df = df.drop(columns=[col for col in drop_columns if col in df.columns])

# 컬럼 타입 변환
for col in ["irrelevant_score", "fear_score", "greed_score"]:
    if col in df.columns:
        df[col] = df[col].astype("float32")

# UTF-8로 저장
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("✅ 인코딩 변환 및 컬럼 제거 완료: samsung_label.csv로 저장됨")
