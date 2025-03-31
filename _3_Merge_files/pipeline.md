[데이터 파이프라인 구축 지침]
1. 프로젝트 개요
- 목적: 종목별(삼성전자, SK하이닉스, 애플, 엔비디아) 투자 심리 지표 제공
- 방법: KcBERT 기반 감성 분석 모델을 통해 공포/중립/탐욕 점수 산출
- 확장성: 수십 종목으로 확장 가능한 구조 설계

2. 기술 스택
- 데이터 처리: Hadoop (HDFS), Spark (전처리 및 배치 처리)
- 워크플로우 관리: Airflow
- 환경 격리 및 배포: Docker
- 백엔드: Django (웹 서비스 제공)

3. 파이프라인 구성 (총 3개)
[1] 학습 파이프라인 (Train)
- 댓글 수집 → 전처리 → 약한 라벨링 → Train/Valid 분할 → KcBERT 모델 학습
- 종목별 모델: kcbert_3class_<stock_name>
- 실행 주기: 비정기 (키워드/데이터 변경 시)
[2] 추론 및 지수 산출 파이프라인 (Inference + Aggregation)
- 새로운 댓글 수집 → 전처리 → 학습된 모델로 감성 점수 예측 → 시간 가중치 기반 지수 산출
- 실행 주기: 정기적 (예: 일간)
[3] 웹서비스 데이터 파이프라인 (Chart/Finance)
- 주가 차트 크롤링 (네이버 금융) → 재무제표 정보 수집 → 파일 저장 또는 DB 적재
- Django는 저장된 데이터를 기반으로 종목 상세 정보 제공 (차트, 지수, 재무제표)

4. 파일 구조 (확장성과 유지보수 고려)
project-root/
├── data/
│   ├── raw/                # 댓글 원본 (samsung.csv 등)
│   ├── preprocess/         # 전처리된 댓글 (samsung_filtered.csv 등)
│   ├── labeling/           # 라벨링 결과 (train/valid)
│   ├── predict/            # 예측 결과 (공포/탐욕 확률)
│   ├── index/              # 최종 감성 지수
│   ├── chart/              # 종목별 주가 차트
│   └── finance/            # 종목별 재무정보
│
├── models/                 # 학습된 KcBERT 모델 저장소
│   └── kcbert_3class_<stock>/
│
├── scripts/                # 학습/추론/전처리 스크립트
│   ├── preprocess.py
│   ├── labeling.py
│   ├── train2.py
│   ├── inference2.py
│   └── aggregate_fear_greed_index.py
│
├── airflow_dags/          # Airflow DAG 파일
│   ├── train_pipeline_dag.py
│   ├── inference_pipeline_dag.py
│   └── finance_pipeline_dag.py
│
├── docker/                # Docker 설정 (Airflow, Spark, Hadoop 등)
│   └── docker-compose.yml
│
└── backend/               # Django 백엔드 서비스
    └── (views, templates, api, static 등)

5. 종목 확장 전략
- stock_name을 파라미터화하여 파이프라인 재사용
- 파일명과 디렉토리는 영어로 통일: samsung, skhynix, apple, nvidia
- Airflow DAG 및 Spark 작업에서 반복 처리 가능하도록 구성

6. 운영 고려사항
- 데이터 유효성 검증 단계 포함 (결측치, 중복 등)
- 파이프라인 실패 대응 및 재시도 설정
- 결과 리포트 및 알림 시스템 (예: Slack, Email)
- 모델 및 데이터 버저닝 적용 (v1, v2 등)
- 향후 사용자 피드백을 활용한 재학습 확장 고려

project-root/
├── data/
│  ├── raw/                 # 종목별 원본 댓글 (ex: samsung.csv)
│  ├── preprocess/          # 정제된 댓글 (samsung_filtered.csv)
│  ├── predict/             # 추론 결과 (samsung_predict_bert.csv)
│  └── index/               # 감성 지수 (samsung_index.csv)
│
├── models/                  # 학습된 KcBERT 모델
│  └── kcbert_3class_<stock>/
│
├── scripts/                 # 추론에 필요한 핵심 스크립트
│  ├── preprocess.py             # 텍스트 정제 및 필터링
│  ├── inference.py             # 감성 점수 예측
│  └── aggregate.py             # 감성 지수 계산
│
├── airflow_dags/           # 추론 전용 DAG 파일
│  └── inference_pipeline_dag.py
│
├── docker/                 # 실행 환경 구성
│  └── docker-compose.yml
│
└── README.md               # 프로젝트 설명 (선택)