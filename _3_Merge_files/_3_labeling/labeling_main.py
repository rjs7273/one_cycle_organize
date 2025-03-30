import pandas as pd
from sklearn.model_selection import train_test_split
from common_labeling import label_dataframe

# stock_list = ["samsung","skhynix","apple","nvidia"]

stock_name = "apple"

# 경로 설정
load_path = f"_0_data/_1_preprocess/{stock_name}_filtered.csv"
save_train = f"_0_data/_2_labeling/{stock_name}_train_3class.csv"
save_valid = f"_0_data/_2_labeling/{stock_name}_valid_3class.csv"
fear_path = "_0_data/_0_raw/fear_keywords.txt"
greed_path = "_0_data/_0_raw/greed_keywords.txt"

# 데이터 로드
df = pd.read_csv(load_path)

# 라벨링
labeled_df = label_dataframe(df, fear_path, greed_path)

# 라벨 분포 확인
print("✅ 전체 라벨 분포 (0=공포, 1=중립, 2=탐욕):")
print(labeled_df["label"].value_counts().sort_index())

# train/valid 분리
train_df, valid_df = train_test_split(
    labeled_df,
    test_size=0.2,
    stratify=labeled_df["label"],
    random_state=42
)

# 저장
save_train = f"_0_data/_2_labeling/{stock_name}_train_3class.csv"
save_valid = f"_0_data/_2_labeling/{stock_name}_valid_3class.csv"
train_df.to_csv(save_train, index=False)
valid_df.to_csv(save_valid, index=False)


print(f"✅ {stock_name} 라벨링 완료! (train: {len(train_df)}, valid: {len(valid_df)})")
