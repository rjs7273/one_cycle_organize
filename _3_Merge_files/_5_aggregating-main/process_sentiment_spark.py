from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg
import os

# ✅ Spark 세션 설정
spark = SparkSession.builder.appName("SentimentAggregation").getOrCreate()

# ✅ 회사 리스트 정의
companies = ["samsung", "apple", "nvidia", "skhynix"]  # 필요한 기업 추가 가능

# ✅ 원본 데이터 & 저장 폴더 경로 설정
raw_data_path = "D:/Project/"
processed_data_path = "./data/processed/"

# ✅ 모든 기업 데이터 처리
for company in companies:
    file_path = f"file:///{raw_data_path}{company}_predict_bert.csv"  # Spark에서 로컬 파일 읽기 형식 적용
    
    print(f"🔍 Checking file: {file_path}")  # 디버깅용 출력
    
    local_file_path = os.path.join(raw_data_path, f"{company}_predict_bert.csv")  # 로컬 파일 존재 여부 확인용
    if not os.path.exists(local_file_path):
        print(f"⚠️ {company} 데이터 파일 없음: {local_file_path}")
        continue

    # 📌 CSV 파일 로드
    df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)

    # 📌 날짜 컬럼 변환
    df = df.withColumn("date", to_date(col("time")))  # "time" 컬럼에서 날짜만 추출

    # 📌 날짜별 감정 비율 평균 계산
    df_daily = df.groupBy("date").agg(
        avg("prob_fear").alias("fear_ratio"),
        avg("prob_neutral").alias("neutral_ratio"),
        avg("prob_greed").alias("greed_ratio")
    )

    # 📌 결과 확인 (상위 5개 데이터)
    print(f"✅ {company} 데이터 처리 완료:")
    df_daily.show(5)

    # 📌 Pandas 변환 후 CSV 저장
    os.makedirs(processed_data_path, exist_ok=True)  # 폴더 없으면 생성
    output_file = os.path.join(processed_data_path, f"{company}_daily_sentiment.csv")
    df_daily.toPandas().to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"💾 {company} 데이터 저장 완료: {output_file}")

print("🎯 모든 기업 감정 분석 완료!")
