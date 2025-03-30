import pandas as pd
from common_preprocess import preprocess_comments

# stock_list = ["samsung","skhynix","apple","nvidia"]

stock_name = "apple"

# 경로 설정
raw_path = f"_0_data/_0_raw/{stock_name}_comments.csv"
keyword_path = "_0_data/_0_raw/political_keywords.txt"
save_path = f"_0_data/_1_preprocess/{stock_name}_filtered.csv"

# CSV 로드
df = pd.read_csv(raw_path)

# 전처리 실행
filtered_df = preprocess_comments(
    df,
    political_path=keyword_path,
    text="Message",         
    timestamp="Updated At"   
)

# 저장
filtered_df.to_csv(save_path, index=False)
print(f"✅ {stock_name} 전처리 완료! 남은 댓글 수: {len(filtered_df)}")
