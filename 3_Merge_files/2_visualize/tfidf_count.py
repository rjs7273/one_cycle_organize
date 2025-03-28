"""

TF-IDF 방식으로 주요 키워드 추출

VSCode 환경에서는 matplotlib GUI가 잘 실행되지 않습니다.
따라서 그래프를 png 파일로 저장하는 형태로 손봤습니다.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


import matplotlib
matplotlib.use('Agg')

# ✅ 1. 전처리된 댓글 데이터 로드
filtered_df = pd.read_csv('../0_data/1_preprocess/삼성전자_filtered.csv')

# ✅ 2. TF-IDF 벡터화 (상위 1000개 단어까지)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(filtered_df['text_tfidf'].fillna(''))

# ✅ 3. TF-IDF 점수 합산 후 상위 30개 추출
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.sum(axis=0).A1  # type: ignore # 희소 행렬을 1차원 배열로 변환
tfidf_df = pd.DataFrame({
    'word': feature_names,
    'score': tfidf_scores
}).sort_values(by='score', ascending=False).head(30)

# ✅ 4. 한글 폰트 설정 (Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# ✅ 5. 시각화
plt.figure(figsize=(10, 8))
sns.barplot(data=tfidf_df, x='score', y='word', color='skyblue')
plt.title('TF-IDF 상위 주식 키워드', fontsize=14)
plt.xlabel('TF-IDF 점수', fontsize=12)
plt.ylabel('키워드', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
# plt.show()
plt.savefig('./chart/tfidf_top30_keywords.png')

