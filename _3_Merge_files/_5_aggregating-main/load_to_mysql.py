import mysql.connector
import pandas as pd
import os

# ✅ MySQL 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Pw',
    'database': 'stock_analysis'
}

# ✅ NaN 값을 None으로 변환하는 함수 (fear_ratio, neutral_ratio, greed_ratio에만 적용)
def replace_nan_for_ratios(val):
    return None if pd.isna(val) else val

# ✅ MySQL 적재 함수
def insert_sentiment_data(file_path, stock_code):
    """
    CSV 파일에서 감정 데이터 로드 후 MySQL 테이블에 INSERT
    """
    # 📌 CSV 데이터 로드
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # 📌 MySQL 연결
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        print(f"📡 {stock_code} 감정 데이터 MySQL INSERT 시작...")

        for index, row in df.iterrows():
            # date 컬럼이 비어있는 경우 건너뛰기
            if pd.isna(row['date']):
                print(f"⚠️ {stock_code} - {index}번째 행의 'date' 값이 비어있어 건너뜁니다.")
                continue

            sql = """
                INSERT INTO sentiment_indicators (stock_code, date, fear_ratio, neutral_ratio, greed_ratio)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    fear_ratio = VALUES(fear_ratio),
                    neutral_ratio = VALUES(neutral_ratio),
                    greed_ratio = VALUES(greed_ratio)
            """
            val = (
                stock_code,
                row['date'],
                replace_nan_for_ratios(row['fear_ratio']),
                replace_nan_for_ratios(row['neutral_ratio']),
                replace_nan_for_ratios(row['greed_ratio'])
            )
            cursor.execute(sql, val)

        # 📌 변경 사항 저장
        conn.commit()
        print(f"✅ {stock_code} 데이터 MySQL 저장 완료!")

    except mysql.connector.Error as err:
        print(f"❌ MySQL 오류 발생: {err}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()
        print("🔌 MySQL 연결 종료")

# ✅ 실행 코드
if __name__ == "__main__":
    # 처리할 회사 리스트
    companies = ["samsung", "apple", "nvidia", "skhynix"]

    for company in companies:
        file_path = f"./data/processed/{company}_daily_sentiment.csv"

        if os.path.exists(file_path):
            insert_sentiment_data(file_path, company)
        else:
            print(f"⚠️ 파일 없음: {file_path}")
