"""
sampling_{}.csv 파일을 불러와
OpenAI에게 레이블링을 지시하고
sample_with_label_{}.csv 파일로 저장하는 코드
"""

from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
import os
from dotenv import load_dotenv

# API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

idx = 10

df = pd.read_csv(f"../0_data/2_sampling/삼성전자_sampling_{idx}.csv")

# 100행 기준 0.7달러 소요. 추천
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_emotion_scores(comment):
    prompt = short_prompt.format(comment=comment)

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content

def clean_response(raw_output):
    import re
    match = re.search(r"{.*}", raw_output, re.DOTALL)
    if match:
        return match.group(0)
    return raw_output

# 결과 저장용 리스트
fear_scores = []
greed_scores = []
neutral_scores = []
reasons = []
# 분석 실행
for i, comment in enumerate(tqdm(df['content'])):
    try:
        output = get_emotion_scores(comment)
        cleaned = clean_response(output)
        print(f"[{i}] GPT 응답:", cleaned)
        if isinstance(cleaned, str):
            parsed = json.loads(cleaned)
        else:
            raise ValueError("GPT 응답이 문자열이 아님")

        fear_scores.append(parsed.get("fear_score", None))
        greed_scores.append(parsed.get("greed_score", None))
        neutral_scores.append(parsed.get("neutral_score", None))
        reasons.append(parsed.get("reason", ""))
    except Exception as e:
        print(f"[{i}] 오류 발생: {e}")
        fear_scores.append(None)
        greed_scores.append(None)
        neutral_scores.append(None)
        reasons.append("오류")

    # 10개마다 중간 저장
    if (i + 1) % 10 == 0:
        df_partial = df.iloc[:i+1].copy()
        df_partial['fear_score'] = fear_scores
        df_partial['greed_score'] = greed_scores
        df_partial['neutral_score'] = neutral_scores
        df_partial['reason'] = reasons

        partial_path = f"../0_data/3_labeling/삼성전자_sample_with_label_partial_{timestamp}.csv"
        df_partial.to_csv(partial_path, index=False)
        print(f">>> 중간 저장 완료: {partial_path}")

# 최종 저장
df['fear_score'] = fear_scores
df['greed_score'] = greed_scores
df['neutral_score'] = neutral_scores
df['reason'] = reasons

df.to_csv(f"../0_data/3_labeling/삼성전자_sample_with_label_{idx}.csv", index=False)