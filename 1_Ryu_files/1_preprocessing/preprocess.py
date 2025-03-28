"""

크롤링한 데이터 전처리 수행하는 코드
 - 데이터 로드 및 통합
 - 테스트 정규화

 단, 네이버 댓글의 경우 오염도가 심하기 때문에
 use_naver 옵션을 False로 두었음.

"""
import pandas as pd
import re
import emoji

# 1. 데이터 로드 및 통합
# use_naver : 네이버 통합할지 여부
def load_and_merge_data(naver_path: str, toss_path: str, use_naver: bool = True) -> pd.DataFrame:
    df_toss = pd.read_csv(toss_path)
    df_toss = df_toss[["content", "timestamp", "platform", "stock_name"]]

    if use_naver:
        df_naver = pd.read_csv(naver_path)
        df_naver["content"] = df_naver["title"].fillna('') + " " + df_naver["content"].fillna('')
        df_naver = df_naver[["content", "timestamp", "platform", "stock_name"]]
        df = pd.concat([df_naver, df_toss], ignore_index=True)
    else:
        df = df_toss.copy()

    return df

# 2. 텍스트 정규화 함수
def normalize_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    text = emoji.replace_emoji(text, replace='')  # 이모지 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)  # 특수문자 제거
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # 반복 문자 축소
    return text.strip()

# 3. 정규화 적용
def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    df["content"] = df["content"].astype(str).apply(normalize_text)
    return df

# 4. 전체 전처리 파이프라인 실행
def preprocess_pipeline(naver_path: str, toss_path: str, tau: int = 86400, use_naver: bool = False) -> pd.DataFrame:
    df = load_and_merge_data(naver_path, toss_path, use_naver)
    df = apply_normalization(df)
    return df


if __name__ == "__main__":
    naver_file = "../0_data/0_raw/naver_삼성전자.csv"
    toss_file = "../0_data/0_raw/toss_삼성전자.csv"

    df_processed = preprocess_pipeline(naver_file, toss_file)
    df_processed.to_csv('../0_data/1_preprocessed/삼성전자_preprocess.csv', index=False, encoding='utf-8-sig')

