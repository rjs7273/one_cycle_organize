"""
sampling_{}.csv 파일을 불러와
OpenAI에게 레이블링을 지시하고
sample_with_label_{}.csv 파일로 저장하는 코드
"""
from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import argparse
import os
from dotenv import load_dotenv
import re

# API 키 설정
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 종목명 설정
parser = argparse.ArgumentParser(description="OpenAI 기반 감정 레이블링")
parser.add_argument("--stock_name", type=str, default="samsung", help="종목명 (예: apple, samsung 등)")
args = parser.parse_args()
stock_name = args.stock_name.lower()

# 경로 설정
base_dir = Path(__file__).resolve().parent
input_path = base_dir / f"../data/_2_sampled/{stock_name}_1000_sample.csv"
output_dir = base_dir / "../data/_3_labeled"
output_dir.mkdir(parents=True, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(input_path)

# 프롬프트 정의
short_prompt = """
주식 커뮤니티 댓글을 다음 세 감정 기준에 따라 분석하라. 공포(Fear), 탐욕(Greed), 중립(Neutral)의 감정 정도를 각각 0~1 사이 소수점 둘째 자리까지 수치로 출력하고, 세 값의 합은 1이어야 한다.

- 공포: 하락, 손실, 후회, 분노, 체념 등 부정 감정 (예: "손절", "또 물림", "하락", "장난질")
- 탐욕: 과도한 기대, 흥분, 자신감, 공격적 매수 심리 (예: "가즈아", "대박", "급등", "들어간다")
- 중립: 감정이 뚜렷하지 않고, 질문, 정보 전달, 고민 또는 관망 태도 (예: "들어갈까?", "혼조세 유지", "고민되네")

문장을 분석하라:
"{comment}"

출력 형식 (예시, JSON):
{{"fear_score": 0.70, "greed_score": 0.10, "neutral_score": 0.20, "reason": "비속어와 손실 언급으로 공포 비중이 높음"}}
"""

def get_emotion_scores(comment):
    prompt = short_prompt.format(comment=comment)
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def clean_response(raw_output):
    match = re.search(r"{.*}", raw_output, re.DOTALL)
    if match:
        return match.group(0)
    return raw_output

# 결과 누적용 리스트
batch_size = 10
batch_data = []

# 분석 실행
for i, comment in enumerate(tqdm(df['Message'])):
    batch_idx = i // batch_size
    batch_path = output_dir / f"backup/{stock_name}_label_{batch_idx}.csv"

    # 해당 배치가 이미 저장돼 있다면 건너뜀
    if batch_path.exists():
        # print(f"[{i}] 이미 처리됨 → 건너뜀: {batch_path.name}")
        continue

    try:
        output = get_emotion_scores(comment)
        cleaned = clean_response(output)
        print(f"[{i}] GPT 응답:", cleaned)

        parsed = json.loads(cleaned) if isinstance(cleaned, str) else {}
        row = {
            "Message": comment,
            "fear_score": parsed.get("fear_score", None),
            "greed_score": parsed.get("greed_score", None),
            "neutral_score": parsed.get("neutral_score", None),
            "reason": parsed.get("reason", ""),
        }
    except Exception as e:
        print(f"[{i}] 오류 발생: {e}")
        row = {
            "Message": comment,
            "fear_score": None,
            "greed_score": None,
            "neutral_score": None,
            "reason": "오류",
        }

    batch_data.append(row)

    # 배치 단위로 저장
    if (i + 1) % batch_size == 0 or (i + 1) == len(df):
        df_batch = pd.DataFrame(batch_data)
        df_batch.to_csv(batch_path, index=False, encoding="utf-8-sig")
        print(f">>> 저장됨: {batch_path}")
        batch_data = []


# 모든 partial CSV 병합
partial_files = sorted(output_dir.glob(f"backup/{stock_name}_label_*.csv"))
merged_df = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)

# 최종 저장
final_path = output_dir / f"{stock_name}_label.csv"
merged_df.to_csv(final_path, index=False, encoding="utf-8-sig")
print(f"[✔] 최종 저장 완료: {final_path}")
