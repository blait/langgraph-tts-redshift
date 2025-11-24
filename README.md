# Redshift 자연어 쿼리 분석 시스템

이 프로젝트는 자연어를 사용하여 Amazon Redshift 데이터를 쿼리하고 분석하는 시스템입니다. LangGraph를 사용하여 워크플로우를 구성하고, Bedrock의 Claude 3.7을 활용하여 자연어 처리를 수행합니다.

<img width="1315" height="399" alt="image" src="https://github.com/user-attachments/assets/e73c4c20-3161-4e7a-910b-38286cd4f11a" />

<img width="1312" height="520" alt="image" src="https://github.com/user-attachments/assets/e1b68029-c3c9-43c1-a312-9f9aff548014" />



## 주요 기능

- 자연어 요청을 Redshift SQL로 변환
- SQL 검증 및 실행
- 결과 검증 및 인사이트 생성
- 오류 발생 시 자동 재시도 및 개선

## 시스템 플로우

1. 사용자 분석 요청 (자연어)
2. 요청에 대한 Redshift SQL 생성
3. SQL 검증 (Redshift EXPLAIN 활용)
4. SQL 실행 및 결과 수신
5. 결과와 SQL로 사용자 요청 충족 여부 검증
6. 데이터와 SQL 기반으로 분석 인사이트 생성

## 설치 및 실행 방법

1. 필요한 패키지 설치:
   ```
   python3 -m venv venv
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. `.env` 파일 설정:
   ```
   cp .env.example .env
   ```
   그리고 `.env` 파일에 Redshift 연결 정보와 AWS 자격 증명을 입력합니다.

3. 애플리케이션 실행:
   ```
   streamlit run app.py
   ```

## 사용 방법

1. 웹 브라우저에서 Streamlit 애플리케이션에 접속합니다.
2. 채팅 인터페이스에 분석하고 싶은 내용을 자연어로 입력합니다.
3. 시스템이 자동으로 SQL을 생성하고, 실행한 후, 결과에 대한 인사이트를 제공합니다.

## 기술 스택

- Streamlit: 사용자 인터페이스
- LangGraph: 워크플로우 관리
- Amazon Bedrock (Claude 3.7): 자연어 처리
- Amazon Redshift: 데이터 저장 및 쿼리
- psycopg2: Redshift 연결
- pandas: 데이터 처리
