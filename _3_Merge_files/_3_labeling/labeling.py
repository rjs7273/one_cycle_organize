import pandas as pd
import re

from sklearn.model_selection import train_test_split

# ✅ 1. 전처리된 댓글 데이터 로드
filtered_df = pd.read_csv('../0_data/1_preprocess/삼성전자_filtered.csv')
texts = filtered_df['text'].fillna('')

# ✅ 2. 감정 키워드 불러오기
with open('../0_data/0_raw/fear_keywords.txt', 'r', encoding='utf-8') as f:
    fear_keywords = set(line.strip() for line in f if line.strip())

with open('../0_data/0_raw/greed_keywords.txt', 'r', encoding='utf-8') as f:
    greed_keywords = set(line.strip() for line in f if line.strip())

# ✅ 3. 텍스트 정제 함수
def clean_text_for_labeling(text):
    return re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", " ", str(text)).lower()

# ✅ 4. 약한 라벨링 함수 (공포=0, 중립=1, 탐욕=2)
def weak_label_by_count(text):
    text_clean = clean_text_for_labeling(text)
    fear_count = sum(keyword in text_clean for keyword in fear_keywords)
    greed_count = sum(keyword in text_clean for keyword in greed_keywords)

    if fear_count > greed_count:
        return 0  # 공포
    elif greed_count > fear_count:
        return 2  # 탐욕
    elif fear_count == greed_count and fear_count > 0:
        return 1  # 중립
    else:
        return 1  # 중립

# ✅ 5. 약한 라벨링 적용
filtered_df['label'] = texts.apply(weak_label_by_count)

# ✅ 6. 라벨 분포 확인
print("✅ 전체 라벨 분포 (0=공포, 1=중립, 2=탐욕):")
print(filtered_df['label'].value_counts().sort_index())

# ✅ 7. 텍스트와 라벨만 추출
train_data = filtered_df[["text", "label"]].copy()

# ✅ 8. stratify로 클래스 비율 유지하며 분할
train_df, valid_df = train_test_split(
    train_data,
    test_size=0.2,
    stratify=train_data["label"],
    random_state=42
)

# ✅ 9. 저장
train_df.to_csv("../0_data/2_labeling/삼성전자_train_3class.csv", index=False)
valid_df.to_csv("../0_data/2_labeling/삼성전자_valid_3class.csv", index=False)

# ✅ 10. 확인
print("Train 분포:\n", train_df["label"].value_counts())
print("Valid 분포:\n", valid_df["label"].value_counts())
