"""
전처리된 댓글 데이터에서 1000개를 무작위로 샘플링하여 저장하는 코드
"""
import pandas as pd
from pathlib import Path
import argparse

if __name__ == "__main__":
    # 종목명 입력 (기본값: samsung)
    parser = argparse.ArgumentParser(description="댓글 데이터 샘플링")
    parser.add_argument("--stock_name", type=str, default="samsung", help="종목명 (예: apple, samsung 등)")
    args = parser.parse_args()
    stock_name = args.stock_name

    # 현재 스크립트 기준 경로 설정
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / f"../data/_1_preprocessed/{stock_name}_preprocessed.csv"
    output_path = base_dir / f"../data/_2_sampled/{stock_name}_1000_sample.csv"

    # 데이터 로드 및 샘플링
    df = pd.read_csv(input_path)
    sampled_df = df.sample(n=1000, random_state=42).reset_index(drop=True)

    # 저장
    sampled_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[✔] 샘플링 완료: {output_path}")
