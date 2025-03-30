import re
import pandas as pd

# 1. 필터링 함수
def load_keyword_file(path: str, as_set: bool = False):
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f if line.strip()]
        return set(lines) if as_set else lines

def contains_political(text: str, keywords: list[str]) -> bool:
    text = re.sub(r'\s+', '', str(text))
    return any(keyword in text for keyword in keywords)

def contains_url(text: str) -> bool:
    return bool(re.search(r'https?://|www\.|\.com|\.kr', str(text)))

# 2. 전처리 함수
def clean_for_tfidf(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_for_bert(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s!?.,ㄱ-ㅎㅏ-ㅣ가-힣]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# 3. 메인 전처리 함수
def preprocess_comments(df: pd.DataFrame, political_path: str, text: str, timestamp: str) -> pd.DataFrame:
    df = df.copy()
    political_keywords = load_keyword_file(political_path)

    def process_row(row):
        text_val = row[text]
        time_val = row[timestamp]
        if pd.isna(text_val) or pd.isna(time_val):
            return None
        if contains_political(text_val, political_keywords) or contains_url(text_val): # type: ignore
            return None
        return pd.Series({
            "raw_text": text_val,
            "text_tfidf": clean_for_tfidf(text_val),
            "text_bert": clean_for_bert(text_val),
            "time": time_val
        })

    processed = df[[text, timestamp]].apply(process_row, axis=1) # type: ignore
    return processed.dropna().reset_index(drop=True)
