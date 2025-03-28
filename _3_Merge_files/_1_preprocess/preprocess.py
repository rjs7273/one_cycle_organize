import pandas as pd
import re

#  1. 데이터 로드
df = pd.read_csv('../0_data/0_raw/toss_삼성전자.csv')

#  2. 정치/혐오 키워드 로드
with open('../0_data/0_raw/political_keywords.txt', 'r', encoding='utf-8') as f:
    political_keywords = [line.strip() for line in f if line.strip()]

#  3. 필터링 함수
def contains_political(text):
    text = re.sub(r'\s+', '', str(text))
    return any(keyword in text for keyword in political_keywords)

def contains_url(text):
    return bool(re.search(r'https?://|www\.|\.com|\.kr', str(text)))

#  4. 전처리 함수
def clean_for_tfidf(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_for_bert(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s!?.,ㄱ-ㅎㅏ-ㅣ가-힣]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

#  5. 통합 전처리 함수
def preprocess_row(row):
    text = row['content']
    timestamp = row['timestamp']

    # 필수 컬럼이 비어있으면 제거
    if pd.isna(text) or pd.isna(timestamp):
        return None, None, None, None
    if contains_political(text) or contains_url(text):
        return None, None, None, None

    return text, clean_for_tfidf(text), clean_for_bert(text), timestamp

#  6. 적용 및 저장
processed = df[['content', 'timestamp']].apply(preprocess_row, axis=1, result_type='expand')
processed.columns = ['text', 'text_tfidf', 'text_bert', 'time']
filtered_df = processed.dropna().reset_index(drop=True)
filtered_df.to_csv('../0_data/1_preprocess/삼성전자_filtered.csv', index=False)

print(f"✅ 최종 필터링 및 전처리 완료! 남은 댓글 수: {len(filtered_df)}")
