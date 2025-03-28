"""
testing_with_labels.csv 데이터를 읽어와
가중치에 따라 점수를 산정, 그 시점의 예상 공포/탐욕 점수 제공하는 코드
"""

import pandas as pd

# 파일 경로
file_path = "../0_data/4_test_data/삼성전자_testing_with_labels_5.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# weight가 0.3679 이상인 댓글만 필터링
filtered_df = df[df['weight'] >= 0.3679].copy()

# 개별 점수 계산: 공포=0배수, 중립=50배수, 탐욕=100배수
filtered_df['individual_score'] = (
    filtered_df['fear_score'] * 0 +
    filtered_df['neutral_score'] * 50 +
    filtered_df['greed_score'] * 100
)

# weight를 반영한 점수
filtered_df['weighted_score'] = filtered_df['individual_score'] * filtered_df['weight']

# 최종 점수 계산
total_weighted_score = filtered_df['weighted_score'].sum()
total_weight = filtered_df['weight'].sum()

final_score = total_weighted_score / total_weight if total_weight > 0 else 0

# 결과 출력
print(f"커뮤니티 감성 지수 (공포/탐욕 점수): {final_score:.2f}")
