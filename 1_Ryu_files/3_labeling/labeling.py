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

# 100행 기준 4달러 소요
long_prompt = """
다음 기준에 따라 입력 문장에서 나타나는 공포(Fear)와 탐욕(Greed), 중립(Neutral)의 감정 정도를 각각 0과 1 사이의 소숫점 둘째 자리까지의 수치로 출력하라. 공포/탐욕/중립 세 수치의 합은 1이어야 한다.
입력받은 문장은 주식 커뮤니티 사이트 이용자들의 댓글 내용이다.

1. 공포 (Fear)
- 정의: 주식의 하락 또는 손해에 대한 반응으로, 투자자가 불안, 좌절, 분노, 체념, 공황 등의 감정을 표현하는 상태. 현금화 욕구가 강하거나, 투자 판단을 후회하거나, 주식 시장에 대한 불신이 드러남.
- 표현 특징:
· 어조: 비관적, 짜증, 분노, 체념
· 감정 대상: 본인(손절, 후회), 종목(비판), 시장(불신)
· 형태: 부정형 표현, 비속어, 체념조, 감탄사 또는 한탄 표현 사용
· 주요 키워드: 손절, 또 물림, 무능, 도망쳐, 하락, 후회, 폭락, 나락, 장난질
- 판단 팁:
· 정보 전달처럼 보이나 감정이 내포된 경우 많음. 예: "아침부터 체결강도 안 좋더라니" → 불만 포함
· 욕설이나 비속어가 포함된 문장은 대부분 공포로 간주

2. 탐욕 (Greed)
- 정의: 종목의 상승에 대한 과도한 기대감, 흥분, 자신감, 공격적 매수 심리가 드러난 상태. 낙관적 전망과 함께 매수 충동이 표현됨. 리스크보다 기회에 집중하는 태도.
- 표현 특징:
· 어조: 흥분, 긍정, 환호, 응원
· 감정 대상: 종목, 시장, 자신
· 형태: 강조형 어미, 느낌표, 유행어(가즈아), 인터넷 은어 사용
· 주요 키워드: 대박, 가즈아, 상한가, 급등, 날아간다, 불붙었다, 들어간다
- 판단 팁:
· 단순한 긍정 표현이라도 담담하거나 정보 중심이면 탐욕 수치는 낮음
· 감탄사, 느낌표, 강조가 많은 문장은 탐욕 수치가 높음
· 과한 자신감이나 흥분이 핵심 판단 기준

3. 중립 (Neutral)
- 정의: 감정이 명확하게 드러나지 않고, 투자 판단에 대한 고민, 질문, 정보 공유가 주된 내용. 불확실성 하의 판단 유보 혹은 의사결정 준비 상태. 주가에 대한 예측 없이 관망하는 태도.
- 표현 특징:
· 어조: 이성적, 분석적, 진지한 고민
· 감정 대상: 없음 (혹은 자기 성찰)
· 형태: 의문형, 추측, 조건절 문장 많음
· 주요 키워드: 고민, 궁금, 들어갈까?, 정신차렸나, 어떻게 될까
- 판단 팁:
· 부정과 혼동되지 않도록 감정 유무 판단 중요
· 중립은 ‘판단을 미루고 있는 상태’에 가깝다는 점 기억

다음 문장을 분석하라:
"{comment}"

출력 형식 (JSON):
{{"fear_score": 0.75, "greed_score": 0.0, "neutral_score": 0.25, "reason": "비관적인 어조와 특정 기술의 실패에 대한 언급으로 공포가 주를 이루나, 구체적인 수치 제시로 일부 중립적 정보 공유의 성격도 포함"}}

출력은 마크다운 코드블럭 없이, 순수 JSON만 출력하라.
"""

# 100행 기준 0.7달러 소요. 추천
short_prompt = """
주식 커뮤니티 댓글을 다음 세 감정 기준에 따라 분석하라. 공포(Fear), 탐욕(Greed), 중립(Neutral)의 감정 정도를 각각 0~1 사이 소수점 둘째 자리까지 수치로 출력하고, 세 값의 합은 반드시 1이어야 한다.

- 공포: 하락, 손실, 후회, 분노, 체념 등 부정 감정 (예: "손절", "또 물림", "하락", "장난질")
- 탐욕: 과도한 기대, 흥분, 자신감, 공격적 매수 심리 (예: "가즈아", "대박", "급등", "들어간다")
- 중립: 감정이 뚜렷하지 않고, 질문, 정보 전달, 고민 또는 관망 태도 (예: "들어갈까?", "혼조세 유지", "고민되네")

문장을 분석하라:
"{comment}"

출력 형식 (예시, JSON):
{{"fear_score": 0.70, "greed_score": 0.10, "neutral_score": 0.20, "reason": "비속어와 손실 언급으로 공포 비중이 높음"}}

출력은 마크다운 코드블럭 없이, 위와 같은 JSON 형식으로만 하라.

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


# 결과 저장
df['fear_score'] = fear_scores
df['greed_score'] = greed_scores
df['neutral_score'] = neutral_scores
df['reason'] = reasons

df.to_csv(f"../0_data/3_labeling/삼성전자_sample_with_label_{idx}.csv", index=False)