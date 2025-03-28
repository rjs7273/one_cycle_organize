"""
sample_with_label_{}.csv 10개 파일을 병합해
sample_with_label_1_to_10.csv로 만드는 코드
"""

import pandas as pd
import os

# 통합할 파일 번호 리스트
file_numbers = range(1, 11)

# 병합한 데이터 저장할 리스트
merged_dfs = []

# 파일 경로 템플릿
input_path_template = "../0_data/3_labeling/삼성전자_sample_with_label_{}.csv"

# 각 파일을 읽어서 리스트에 추가
for num in file_numbers:
    file_path = input_path_template.format(num)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        merged_dfs.append(df)
        print(f"불러오기 완료: {file_path}")
    else:
        print(f"파일 없음: {file_path}")

# 데이터프레임 병합
merged_df = pd.concat(merged_dfs, ignore_index=True)

# 저장
output_path = "../0_data/3_labeling/삼성전자_sample_with_label_1_to_10.csv"
merged_df.to_csv(output_path, index=False)
print(f">>> 저장 완료: {output_path}")
