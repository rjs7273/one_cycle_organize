"""
크롤링한 데이터 전처리 수행하는 코드
"""
import pandas as pd
import re
import emoji
from pathlib import Path
import argparse


# 텍스트 정규화 함수
def normalize_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    text = emoji.replace_emoji(text, replace='')  # 이모지 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)  # 특수문자 제거
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # 반복 문자 축소
    return text.strip()

if __name__ == "__main__":
    # argparse를 이용해 종목명을 입력받음
    parser = argparse.ArgumentParser(description="크롤링한 댓글 데이터 전처리")
    parser.add_argument("--stock_name", type=str, default="samsung", help="종목명 (예: apple, samsung 등)")
    args = parser.parse_args()

    stock_name = args.stock_name.lower()

    # 현재 스크립트 기준 상대 경로 설정
    base_dir = Path(__file__).resolve().parent
    raw_file = base_dir / f"../data/_0_raw/{stock_name}_comments.csv"
    output_file = base_dir / f"../data/_1_preprocessed/{stock_name}_preprocessed.csv"

    # 데이터 로드 및 전처리
    df = pd.read_csv(raw_file)[["Comment ID", "Message", "Updated At", "Stock Name"]]
    df["Message"] = df["Message"].astype(str).apply(normalize_text)

    # 출력 폴더 생성 및 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
