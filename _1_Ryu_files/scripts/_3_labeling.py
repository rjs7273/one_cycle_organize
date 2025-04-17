"""
수정사항
샘플링 10000개로 수정
프롬프트 수정 (irrelevant_score에 대한 정의 명확하게게)
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
output_dir = base_dir / "../data/_3_labeled"
backup_dir = output_dir / "backup"
backup_dir.mkdir(parents=True, exist_ok=True)

# ✅ 4. 데이터 불러오기
df = pd.read_csv(input_path)

# ✅ 5. 최적 프롬프트
short_prompt = """
주식 커뮤니티 댓글을 감정 분석하라.

1. 문장이 투자 심리 분석에 적절한지 판단하라.
   - 투자 심리와 관련된 감정(불안, 기대 등)이 명확한 정도에 따라 irrelevant_score를 산정하라. 낮을수록 투자 심리가 명확하다고 판단한다.
   - 분석 가능성은 다양한 분포(예 : 0.12, 0.48, 0.78)로 표현하고 소수점 둘째 자리까지 반드시시 표현하라.
   - 예시:
   - "와 드디어 떡상 시작이냐" → irrelevant_score: 0.10 (투자 심리 명확)
   - "갑자기 배가 아프네 아이고배야 화장실이나 가야징" → irrelevant_score: 0.95 (투자와 무관한 표현)
   - "단타 치는분들이랑 수익 노리는 분들 같은데요, 5년그래프보면 고양이귀" → irrelevant_score: 0.60 (맥락이 불분명하고 감정도 모호)

2. fear_score와 greed_score를 0~1 사이 반드시 소수점 둘째 자리까지 부여하라.
   - fear_score와 greed_score의 합은 반드시 1이어야 한다.
   - 공포(fear): 손실, 하락에 대한 불안, 체념, 후회, 분노, 시장 불신, 투자 포기 또는 관망 심리 등
   - 탐욕(greed): 급등 기대, 과도한 낙관, 흥분, 공격적 매수 심리, 무리한 확신 등 (“가즈아”, “떡상” 같은 표현이 해당됨)

출력은 반드시 다음 JSON 형식으로, 모든 key를 누락 없이 포함하라.  
"reason"은 감정 판단의 핵심 단서만 요약하여 반드시 1문장 이내로 작성하라. 
{
  "irrelevant_score": float, "fear_score": float, "greed_score": float, "reason": "비속어와 손실 언급으로 공포 비중이 높음"
}

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

# ✅ 8. 가장 마지막 저장된 batch index 확인
existing_batches = list(backup_dir.glob(f"{stock_name}_label_*.csv"))
existing_indices = {int(f.stem.split("_")[-1]) for f in existing_batches}
print(f"✅ 기존 저장된 배치 인덱스: {sorted(existing_indices)}")

# ✅ 9. 분석 실행
batch_size = 10
total = len(df)
max_batch_idx = (total - 1) // batch_size

for batch_idx in range(max_batch_idx + 1):
    if batch_idx in existing_indices:
        continue

    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, total)
    batch_comments = df.iloc[batch_start:batch_end]["Message"]

    batch_data = []
    for i, comment in enumerate(batch_comments, start=batch_start):
        try:
            output = get_emotion_scores(comment)
            cleaned = clean_response(output)
            print(f"[{i}] GPT 응답:", cleaned)

            parsed = json.loads(cleaned) if isinstance(cleaned, str) else {}
            row = {
                "Message": comment,
                "irrelevant_score": parsed.get("irrelevant_score", None),
                "fear_score": parsed.get("fear_score", None),
                "greed_score": parsed.get("greed_score", None),
                "reason": parsed.get("reason", ""),
            }
        except Exception as e:
            print(f"[{i}] 오류 발생: {e}")
            row = {
                "Message": comment,
                "irrelevant_score": None,
                "fear_score": None,
                "greed_score": None,
                "reason": "오류",
            }
        batch_data.append(row)

    batch_df = pd.DataFrame(batch_data)
    batch_path = backup_dir / f"{stock_name}_label_{batch_idx}.csv"
    batch_df.to_csv(batch_path, index=False, encoding="utf-8-sig")
    print(f">>> 저장됨: {batch_path}")

# ✅ 10. 최종 병합 및 저장
partial_files = sorted(backup_dir.glob(f"{stock_name}_label_*.csv"))
merged_df = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)
final_path = output_dir / f"{stock_name}_label_1000.csv"
merged_df.to_csv(final_path, index=False, encoding="utf-8-sig")
print(f"[✔] 최종 저장 완료: {final_path}")
