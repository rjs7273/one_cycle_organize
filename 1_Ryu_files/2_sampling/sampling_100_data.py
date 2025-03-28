"""
./preprocess/삼성전자_preprocess.py에서 
중복 없이 무작위로 100개씩 샘플링된
50개의 CSV 파일을 생성하는 코드
"""

import pandas as pd
import random
import os

# 입력 파일 경로
input_path = '../0_data/1_preprocessed/삼성전자_preprocess.csv'

# 출력 디렉토리
output_dir = '../0_data/2_sampled'
os.makedirs(output_dir, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(input_path)

# 중복 제거를 위해 전체 데이터에서 임의의 1000개 행 샘플링
sampled_df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# 각 100개씩 분할하고 저장
for number in range(1, 51):
    start_idx = (number - 1) * 100
    end_idx = number * 100
    sub_df = sampled_df.iloc[start_idx:end_idx].copy()
    sub_df['label'] = ''  # label 컬럼 추가 (빈 문자열)
    
    output_path = os.path.join(output_dir, f'삼성전자_sampling_{number}.csv')
    sub_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("샘플링 및 저장 완료.")
