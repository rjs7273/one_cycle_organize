from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, hour, year, lag, month
from pyspark.sql.window import Window
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import mysql.connector

# ✅ 환경 설정 함수
def setup():
    """
    - 한글 폰트 설정
    - 결과 저장 폴더 생성
    - Spark 세션 생성 및 설정
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs('./chart', exist_ok=True)

    spark = (SparkSession.builder
                .master("local")
                .appName("SentimentAggregation")
                .config("spark.ui.showConsoleProgress", "true")
                .getOrCreate())
    spark.sparkContext.setLogLevel("INFO")
    return spark

# ✅ 데이터 로드 및 전처리 함수
def load_and_preprocess(spark, file_path):
    """
    - CSV 파일을 읽어와 DataFrame으로 변환
    - 날짜, 시간, 연도 컬럼 추가
    - 공포탐욕지수 계산
    """
    df = spark.read.option("header", True).option("encoding", "UTF-8").csv(file_path, inferSchema=True)
    df = df.withColumn("date", to_date(col("time")))
    df = df.withColumn("hour", hour(col("time")))
    df = df.withColumn("year", year(col("date")))
    df = df.withColumn("공포탐욕지수", col("prob_greed") * 100)
    return df

# ✅ 공포탐욕지수 평균 계산 및 저장 함수
def calculate_fear_greed(df, company):
    """
    - 시간대별 평균 공포탐욕지수 계산 및 저장
    - 월간 평균 공포탐욕지수 계산 및 저장
    """
    df_hourly = df.groupBy("year", "hour").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    df_hourly.toPandas().to_csv(f"./chart/{company}_hourly_feargreed_score_bert.csv", index=False, encoding="utf-8-sig")

    df = df.withColumn("month", col("date").substr(1, 7))  #<-MM 형태 변환
    df_monthly = df.groupBy("month").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    df_monthly_pandas = df_monthly.toPandas()
    df_monthly_pandas = df_monthly_pandas.dropna(subset=["month"])  # 결측값 제거
    df_monthly_pandas["month"] = df_monthly_pandas["month"].astype(str)  # 문자열 변환
    df_monthly_pandas = df_monthly_pandas.sort_values(by="month").reset_index(drop=True)
    df_monthly_pandas.to_csv(f"./chart/{company}_monthly_feargreed_score_bert.csv", index=False, encoding="utf-8-sig")
    return df_monthly_pandas

# ✅ 공포탐욕지수 변화율 계산 함수
def calculate_change_rate(df, company):
    """
    - 시간대별 공포탐욕지수 변화율 계산
    - 결과 CSV 저장
    """
    window_spec = Window.partitionBy("year").orderBy("hour")
    df = df.withColumn("feargreed_diff", col("공포탐욕지수") - lag(col("공포탐욕지수"), 1).over(window_spec))
    df_change_rate = df.groupBy("year", "hour").agg(avg("feargreed_diff").alias("변화율"))
    df_change_rate_pandas = df_change_rate.toPandas()
    df_change_rate_pandas = df_change_rate_pandas.sort_values(by=["year", "hour"]).reset_index(drop=True)
    df_change_rate_pandas.to_csv(f"./chart/{company}_feargreed_change_rate.csv", index=False, encoding="utf-8-sig")
    return df_change_rate_pandas

# ✅ 이동 평균 분석 함수 (주석 처리)
# def calculate_moving_average(df_monthly_pandas, company):
#     """
#     - 단기(7일) 및 장기(30일) 이동 평균 계산
#     - 골든크로스 및 데드크로스 감지
#     """
#     df_monthly_pandas["단기_이동평균"] = df_monthly_pandas["평균_공포탐욕지수"].rolling(window=7).mean()
#     df_monthly_pandas["장기_이동평균"] = df_monthly_pandas["평균_공포탐욕지수"].rolling(window=30).mean()
#     df_monthly_pandas.to_csv(f"./chart/{company}_moving_average.csv", index=False, encoding="utf-8-sig")
#     return df_monthly_pandas

# ✅ 클러스터 분석 함수 (기존 주석 처리 유지)
# def cluster_analysis(df_monthly_pandas, company, n_clusters=3):
#     """
#     - K-Means를 활용한 감성 데이터 클러스터링
#     """
#     df_numeric = df_monthly_pandas.drop(columns=["month"]).dropna()
#
#     # 데이터 개수 확인
#     if len(df_numeric) < n_clusters:
#         print(f"⚠️ {company} 데이터에서 클러스터링을 수행할 수 없습니다 (데이터 개수 부족: {len(df_numeric)}개).")
#         df_monthly_pandas["클러스터"] = np.nan
#         return df_monthly_pandas
#
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(df_numeric)
#
#     # 클러스터 결과를 원본 DataFrame에 병합
#     df_clusters = pd.DataFrame({"클러스터": clusters}, index=df_numeric.index)
#     df_monthly_pandas = pd.concat([df_monthly_pandas, df_clusters], axis=1)
#
#     df_monthly_pandas.to_csv(f"./chart/{company}_cluster_analysis.csv", index=False, encoding="utf-8-sig")
#     return df_monthly_pandas

# ✅ 시각화 함수
def save_plots(df_change_rate_pandas, df_monthly_pandas, company):
    """
    - 공포탐욕지수 변화율 및 월간 평균 그래프 저장
    """
    years = df_change_rate_pandas["year"].unique()

    for year in years:
        df_yearly = df_change_rate_pandas[df_change_rate_pandas["year"] == year]

        plt.figure(figsize=(12, 6))
        plt.plot(df_yearly["hour"], df_yearly["변화율"], marker='o', linestyle='-', color='red', label=f'{year}년 공포탐욕지수 변화율')
        plt.axhline(0, color='gray', linestyle='--', label='기준선')
        plt.title(f"{company} {year}년 시간대별 공포탐욕지수 변화율")
        plt.xlabel("시간")
        plt.ylabel("변화율")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./chart/{company}_{year}_fear_and_greed_change_rate.png")
        plt.close()

    print(f"--- {company} 월간 데이터 확인 ---")
    print(df_monthly_pandas)
    print("------------------------------------")

    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly_pandas["month"], df_monthly_pandas["평균_공포탐욕지수"], marker='o', linestyle='-', color='blue', label='월간 평균 공포탐욕지수')
    plt.title(f"{company} 월간 평균 공포탐욕지수")
    plt.xlabel("월")
    plt.ylabel("평균 공포탐욕지수")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./chart/{company}_monthly_fear_and_greed.png")
    plt.close()

# MySQL에 데이터 삽입하는 함수 (이동 평균 관련 부분 주석 처리)
def insert_to_mysql(cursor, company, df_hourly_pandas, df_monthly_pandas, df_change_rate_pandas):
    # NaN 값을 None으로 변환하는 헬퍼 함수
    def replace_nan(val):
        return None if pd.isna(val) else val

    # 시간별 공포탐욕지수 저장
    df_hourly_pandas_cleaned = df_hourly_pandas.dropna(subset=['year', 'hour'])
    for index, row in df_hourly_pandas_cleaned.iterrows():
        sql = "INSERT INTO hourly_feargreed_bert (company, year, hour, 평균_공포탐욕지수) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE 평균_공포탐욕지수=%s"
        val = (company, int(row['year']), int(row['hour']), replace_nan(row['평균_공포탐욕지수']), replace_nan(row['평균_공포탐욕지수']))
        cursor.execute(sql, val)

    # 월별 공포탐욕지수 저장
    for index, row in df_monthly_pandas.iterrows():
        sql = "INSERT INTO monthly_feargreed_bert (company, month, 평균_공포탐욕지수) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE 평균_공포탐욕지수=%s"
        val = (company, row['month'], replace_nan(row['평균_공포탐욕지수']), replace_nan(row['평균_공포탐욕지수']))
        cursor.execute(sql, val)

    # 시간별 변화율 저장
    df_change_rate_pandas_cleaned = df_change_rate_pandas.dropna(subset=['year', 'hour'])
    for index, row in df_change_rate_pandas_cleaned.iterrows():
        sql = "INSERT INTO feargreed_change_rate (company, year, hour, 변화율) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE 변화율=%s"
        val = (company, int(row['year']), int(row['hour']), replace_nan(row['변화율']), replace_nan(row['변화율']))
        cursor.execute(sql, val)

    # 이동 평균 저장 (주석 처리)
    # for index, row in df_moving_avg.iterrows():
    #     sql = "INSERT INTO moving_average (company, month, 평균_공포탐욕지수, 단기_이동평균, 장기_이동평균) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE 평균_공포탐욕지수=%s, 단기_이동평균=%s, 장기_이동평균=%s"
    #     val = (
    #         company, row['month'], replace_nan(row['평균_공포탐욕지수']),
    #         replace_nan(row['단기_이동평균']), replace_nan(row['장기_이동평균']),
    #         replace_nan(row['평균_공포탐욕지수']), replace_nan(row['단기_이동평균']), replace_nan(row['장기_이동평균'])
    #     )
    #     cursor.execute(sql, val)

    # 클러스터 분석 결과 저장 (기존 주석 처리 유지)
    # df_cluster = pd.read_csv(f"./chart/{company}_cluster_analysis.csv", encoding="utf-8-sig")
    # for index, row in df_cluster.iterrows():
    #     cluster_value = row['클러스터']
    #     if pd.isna(cluster_value):
    #         sql = "INSERT INTO cluster_analysis (company, month, 평균_공포탐욕지수, 클러스터) VALUES (%s, %s, %s, NULL) ON DUPLICATE KEY UPDATE 평균_공포탐욕지수=%s, 클러스터=NULL"
    #         val = (company, row['month'], row['평균_공포탐욕지수'], row['평균_공포탐욕지수'])
    #     else:
    #         sql = "INSERT INTO cluster_analysis (company, month, 평균_공포탐욕지수, 클러스터) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE 평균_공포탐욕지수=%s, 클러스터=%s"
    #         val = (company, row['month'], row['평균_공포탐욕지수'], int(cluster_value), row['평균_공포탐욕지수'], int(cluster_value))
    #     cursor.execute(sql, val)

# ✅ 실행 코드
spark = setup()

# MySQL 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Pw',
    'database': 'stock_analysis'
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

try:
    print("MySQL 데이터베이스 연결 성공")
    companies = ["samsung", "apple", "nvidia", "skhynix"]
    for company in companies:
        print(f"--- 현재 처리 중인 회사: {company} ---")
        try:
            file_path = f"file:///D:/Project/{company}_predict_bert.csv"
            df = load_and_preprocess(spark, file_path)
            df_monthly_pandas = calculate_fear_greed(df, company)
            df_change_rate_pandas = calculate_change_rate(df, company)
            # df_monthly_pandas = calculate_moving_average(df_monthly_pandas, company) # 이동 평균 함수 호출 주석 처리
            # df_monthly_pandas = cluster_analysis(df_monthly_pandas, company) # 클러스터링 함수 호출 주석 처리
            save_plots(df_change_rate_pandas, df_monthly_pandas, company)

            df_hourly_pandas = df.groupBy("year", "hour").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수")).toPandas()

            insert_to_mysql(cursor, company, df_hourly_pandas, df_monthly_pandas, df_change_rate_pandas)
            conn.commit()
            print(f"{company} 데이터 MySQL 저장 완료")
        except Exception as e:
            print(f"{company} 처리 중 오류 발생: {e}")
            conn.rollback()  # 오류 발생 시 롤백
            continue  # 다음 회사로 진행
except mysql.connector.Error as err:
    print(f"MySQL 오류 발생: {err}")
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL 연결 종료")

spark.stop()
