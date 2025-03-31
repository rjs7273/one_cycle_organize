from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

# DAG 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

def run_mysql_insert_script():
    """
    MySQL 데이터 적재 스크립트를 실행하는 함수
    """
    script_path = "/path/to/your_script.py"  # 실제 경로로 변경해야 함
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

# DAG 정의
dag = DAG(
    'insert_sentiment_data',
    default_args=default_args,
    description='MySQL 감정 데이터 적재',
    schedule_interval='@daily',  # 매일 실행
    catchup=False,
)

# PythonOperator로 스크립트 실행
task_insert_data = PythonOperator(
    task_id='run_mysql_insert',
    python_callable=run_mysql_insert_script,
    dag=dag,
)

task_insert_data
