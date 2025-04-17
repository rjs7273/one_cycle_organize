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

# ✅ 1. API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ✅ 2. 종목명 인자 처리
parser = argparse.ArgumentParser(description="OpenAI 기반 감정 레이블링")
parser.add_argument("--stock_name", type=str, default="samsung", help="종목명 (예: apple, samsung 등)")
args = parser.parse_args()
stock_name = args.stock_name.lower()

# ✅ 3. 파일 경로 설정
base_dir = Path(__file__).resolve().parent
input_path = base_dir / f"../data/_2_sampled/{stock_name}_10000_sample.csv"
output_dir = base_dir / "../data/_3_labeled/10000"
backup_dir = output_dir / "backup"
backup_dir.mkdir(parents=True, exist_ok=True)

# ✅ 4. 데이터 불러오기
df = pd.read_csv(input_path)

# ✅ 5. 최적 프롬프트
short_prompt = """
주식 커뮤니티 댓글을 감정 분석하라.

1. 문장이 투자 심리 분석에 적절한지 판단하라.
   - 투자 심리와 관련된 감정(불안, 기대 등)이 명확하다면, irrelevant_score는 낮고 fear_score 또는 greed_score 중 하나가 높아야 한다.
   - 분석 가능성은 binary가 아니라 irrelevant_score (0.0 ~ 1.0) 으로 표현하라.

2. irrelevant_score가 높아도 fear_score와 greed_score를 0~1 사이 소수점 둘째 자리까지 부여하라.
   - 두 점수의 합은 반드시 1이 되어야하며 감정 강도에 따라 다양한 분포(예: 0.62 / 0.38, 0.51 / 0.49 등)로 표현한다.
   - 각 점수는 0.05 단위에 제한되지 말고 자유롭게 부여하라.

3. 중립 여부는 이 단계에서 판단하지 않는다. 후처리에서 fear/greed 점수를 기준으로 판별할 것이다.

[감정 정의 - 투자 심리 기준]
- 공포(fear): 손실, 하락에 대한 불안, 체념, 후회, 분노, 시장 불신, 투자 포기 또는 관망 심리 등
- 탐욕(greed): 급등 기대, 과도한 낙관, 흥분, 공격적 매수 심리, 무리한 확신 등 (“가즈아”, “떡상” 같은 표현이 해당됨)

출력은 반드시 다음 JSON 형식으로, 모든 key를 누락 없이 포함하라.  
{
  "irrelevant_score": float, "fear_score": float, "greed_score": float}

문장:
"{comment}"
"""

# ✅ 6. GPT 호출 함수
def get_emotion_scores(comment):
    prompt = short_prompt.replace("{comment}", comment)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content

# ✅ 7. 응답 클린업
def clean_response(raw_output):
    match = re.search(r"{.*}", raw_output, re.DOTALL)
    if match:
        return match.group(0)
    return raw_output

# ✅ 8. 분석 실행
batch_size = 10
batch_data = []

for i, comment in enumerate(tqdm(df['Message'])):
    batch_idx = i // batch_size
    batch_path = backup_dir / f"{stock_name}_label_{batch_idx}.csv"

    if batch_path.exists():
        continue

    try:
        output = get_emotion_scores(comment)
        cleaned = clean_response(output)
        print(f"[{i}] GPT 응답:", cleaned)

        parsed = json.loads(cleaned) if isinstance(cleaned, str) else {}
        row = {
            "Message": comment,
            "irrelevant_score": parsed.get("irrelevant_score", None),
            "fear_score": parsed.get("fear_score", None),
            "greed_score": parsed.get("greed_score", None)
        }
    except Exception as e:
        print(f"[{i}] 오류 발생: {e}")
        row = {
            "Message": comment,
            "fear_score": None,
            "greed_score": None,
            "irrelevant_score": None,
        }

    batch_data.append(row)

    if (i + 1) % batch_size == 0 or (i + 1) == len(df):
        df_batch = pd.DataFrame(batch_data)
        df_batch.to_csv(batch_path, index=False, encoding="utf-8-sig")
        print(f">>> 저장됨: {batch_path}")
        batch_data = []

# ✅ 9. 최종 병합 및 저장
partial_files = sorted(backup_dir.glob(f"{stock_name}_label_*.csv"))
merged_df = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)
final_path = output_dir / f"{stock_name}_label_2000.csv"
merged_df.to_csv(final_path, index=False, encoding="utf-8-sig")
print(f"[✔] 최종 저장 완료: {final_path}")
