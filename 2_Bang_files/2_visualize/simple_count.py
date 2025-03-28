"""

단순 등장 횟수 기반 키워드 추출

VSCode 환경에서는 matplotlib GUI가 잘 실행되지 않습니다.
따라서 그래프를 png 파일로 저장하는 형태로 손봤습니다.

"""

import matplotlib
matplotlib.use('Agg')  # Tkinter GUI 백엔드 비활성화

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# ✅ 1. 전처리된 댓글 데이터 로드
filtered_df = pd.read_csv('../0_data/1_preprocess/삼성전자_filtered.csv')
texts = filtered_df['text_tfidf'].fillna('')

# ✅ 2. 텍스트 정제 함수
def clean_text(text):
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", text)  # 한글, 영문, 숫자만 유지
    return text.lower()

# ✅ 3. 전처리 + 단어 리스트 생성
all_words = []
for doc in texts:
    cleaned = clean_text(doc)
    words = cleaned.split()
    all_words.extend(words)

# ✅ 4. 단어 빈도 계산
word_freq = Counter(all_words)
common_words = word_freq.most_common(50)

# ✅ 5. 데이터프레임 변환
freq_df = pd.DataFrame(common_words, columns=['word', 'count'])

# ✅ 6. 폰트 설정 (Windows에서 한글 잘 보이도록)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'AppleGothic' (Mac), 'NanumGothic' (Colab)
plt.rcParams['axes.unicode_minus'] = False

# ✅ 7. 시각화
plt.figure(figsize=(10, 8))
sns.barplot(data=freq_df, x='count', y='word', color='skyblue')
plt.title("단어 등장 빈도 상위 50개 (토스 삼전 댓글)", fontsize=14)
plt.xlabel("등장 횟수", fontsize=12)
plt.ylabel("단어", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

# ✅ 8. 이미지 저장
plt.savefig('./chart/word_frequency_top50.png')
print("✅ 단어 빈도 시각화 저장 완료: word_frequency_top50.png")
