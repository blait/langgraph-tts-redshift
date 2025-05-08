import streamlit as st
import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import psycopg2
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List, Dict, Literal
import json
import os
from dotenv import load_dotenv
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# Redshift 연결 파라미터
redshift_host = os.getenv("REDSHIFT_HOST")
redshift_port = os.getenv("REDSHIFT_PORT")
redshift_dbname = os.getenv("REDSHIFT_DBNAME")
redshift_user = os.getenv("REDSHIFT_USER")
redshift_password = os.getenv("REDSHIFT_PASSWORD")

# Bedrock 클라이언트 초기화 (Claude 3.7)
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-west-2")
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_kwargs={"temperature": 0.2, "max_tokens": 4000}
)

# Redshift 연결 함수
def get_redshift_connection():
    try:
        conn = psycopg2.connect(
            host=redshift_host,
            port=redshift_port,
            dbname=redshift_dbname,
            user=redshift_user,
            password=redshift_password
        )
        return conn
    except Exception as e:
        st.error(f"Redshift 연결 오류: {e}")
        logger.error(f"Redshift 연결 오류: {e}")
        return None

# Redshift에서 테이블 스키마 가져오기
def get_table_schema():
    conn = get_redshift_connection()
    if not conn:
        return "Redshift 연결에 실패했습니다."
    
    cursor = conn.cursor()
    try:
        query = """
        SELECT 
            table_schema, 
            table_name, 
            column_name, 
            data_type 
        FROM 
            information_schema.columns 
        WHERE 
            table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY 
            table_schema, table_name, ordinal_position;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        schema_info = {}
        for row in rows:
            schema, table, column, data_type = row
            if schema not in schema_info:
                schema_info[schema] = {}
            if table not in schema_info[schema]:
                schema_info[schema][table] = []
            schema_info[schema][table].append({"column": column, "data_type": data_type})
        
        schema_text = ""
        for schema in schema_info:
            schema_text += f"스키마: {schema}\n"
            for table in schema_info[schema]:
                schema_text += f"  테이블: {table}\n"
                for col in schema_info[schema][table]:
                    schema_text += f"    컬럼: {col['column']}, 타입: {col['data_type']}\n"
        
        return schema_text
    except Exception as e:
        logger.error(f"스키마 정보 가져오기 오류: {e}")
        return f"스키마 정보 가져오기 오류: {e}"
    finally:
        cursor.close()
        conn.close()

# Redshift ANALYZE를 사용하여 SQL 검증
def validate_sql(sql):
    conn = get_redshift_connection()
    if not conn:
        return False, "Redshift 연결에 실패했습니다."
    
    cursor = conn.cursor()
    try:
        cursor.execute(f"EXPLAIN {sql}")
        return True, "SQL이 유효합니다."
    except Exception as e:
        logger.error(f"SQL 검증 실패: {e}")
        return False, str(e)
    finally:
        cursor.close()
        conn.close()

# SQL 실행 및 결과 가져오기
def execute_sql(sql):
    conn = get_redshift_connection()
    if not conn:
        return None, "Redshift 연결에 실패했습니다."
    
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        return df, None
    except Exception as e:
        logger.error(f"SQL 실행 실패: {e}")
        return None, str(e)
    finally:
        cursor.close()
        conn.close()

# 상태 정의
class GraphState(TypedDict):
    user_request: str
    sql: Optional[str]
    is_valid: Optional[bool]
    results: Optional[List[Dict]]
    execution_successful: Optional[bool]
    verification_passed: Optional[bool]
    verification_message: Optional[str]
    insights: Optional[str]
    error_feedback: Optional[str]
    status: Optional[str]
    retry_count: Optional[int]

# LangGraph 노드
def generate_sql(state: GraphState) -> GraphState:
    # 현재 상태 로깅
    retry_count = state.get("retry_count", 0)
    error_feedback = state.get("error_feedback", "")
    
    logger.info(f"generate_sql 노드 실행 (재시도: {retry_count})")
    if error_feedback:
        logger.info(f"피드백 정보: {error_feedback[:200]}...")
    
    schema = get_table_schema()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 Amazon Redshift SQL 전문가입니다.
        자연어 쿼리를 정확한 SQL 쿼리로 변환하는 것이 당신의 임무입니다.
        제공된 스키마 정보를 사용하여 정확한 SQL을 작성하세요.
        설명이나 마크다운 형식 없이 SQL 쿼리만 반환하세요.
        Redshift에 최적화된 SQL을 작성하세요.
        
        이전 쿼리에 문제가 있었다면, 그 피드백을 참고하여 개선된 쿼리를 작성하세요."""),
        ("user", """스키마 정보:
        {schema}
        
        사용자 요청: {user_request}
        
        {error_feedback}
        
        이 요청에 대한 SQL 쿼리를 생성하세요. 지표계산 법은 아래를 참조하시오.
         **지표 정의 및 계산 방법**:
1. PUR (구매율):
   - 정의: 구매 사용자 수(PU)를 일일 활성 사용자 수(DAU)로 나눈 비율.
   - 계산: SUM(CASE WHEN cat = 'pu' THEN f_value END) / SUM(CASE WHEN cat = 'dau' THEN f_value END)
   - 출력: 소수점 2자리로 반올림 (예: 0.05).
2. PU (구매 사용자 수):
   - 정의: 구매를 수행한 사용자 수.
   - 계산: MAX(CASE WHEN cat = 'pu' THEN f_value END)
   - 출력: 정수.
3. ARPPU (평균 구매 사용자당 매출):
   - 정의: 구매 사용자 1명당 평균 매출.
   - 계산: SUM(CASE WHEN cat = 'amt_total' THEN f_value END) / SUM(CASE WHEN cat = 'pu' THEN f_value END)
   - 출력: 소수점 0자리로 반올림 (예: 200).
4. DAU (일일 활성 사용자 수):
   - 정의: 특정 날짜에 앱 또는 서비스를 사용한 고유 사용자 수.
   - 계산: MAX(CASE WHEN cat = 'dau' THEN f_value END)
   - 출력: 정수.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "schema": schema,
        "user_request": state["user_request"],
        "error_feedback": error_feedback
    })
    
    sql = response.content.strip()
    if sql.startswith("```sql"):
        sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # 재시도 횟수 증가
    new_retry_count = retry_count + 1
    
    logger.info(f"SQL 생성됨 (재시도 {new_retry_count}번째): {sql[:100]}...")
    return {
        "sql": sql, 
        "status": "sql_generated",
        "retry_count": new_retry_count,
        "user_request": state["user_request"]  # 사용자 요청 유지
    }

def validate_sql_node(state: GraphState) -> GraphState:
    logger.info(f"validate_sql_node 노드 실행 (재시도: {state.get('retry_count', 0)})")
    sql = state["sql"]
    is_valid, error_message = validate_sql(sql)
    
    logger.info(f"SQL 검증 결과: {is_valid}")
    
    if is_valid:
        return {
            "is_valid": True, 
            "status": "sql_validated", 
            "sql": sql,
            "user_request": state["user_request"],
            "retry_count": state.get("retry_count", 0)
        }
    else:
        return {
            "is_valid": False,
            "error_feedback": f"SQL 검증 실패: {error_message}. SQL 쿼리를 수정해주세요.",
            "status": "sql_invalid",
            "sql": sql,
            "user_request": state["user_request"],
            "retry_count": state.get("retry_count", 0)
        }

def execute_sql_node(state: GraphState) -> GraphState:
    logger.info(f"execute_sql_node 노드 실행 (재시도: {state.get('retry_count', 0)})")
    sql = state["sql"]
    df, error = execute_sql(sql)
    
    # 기본 상태 유지를 위한 딕셔너리
    result = {
        "sql": sql,
        "user_request": state["user_request"],
        "retry_count": state.get("retry_count", 0),
        "is_valid": state.get("is_valid", False)
    }
    
    if df is not None:
        results = df.to_dict(orient="records")
        logger.info(f"SQL 실행 성공: {len(results)} 행 반환됨")
        result.update({
            "results": results, 
            "execution_successful": True, 
            "status": "sql_executed"
        })
        return result
    else:
        logger.error(f"SQL 실행 실패: {error}")
        result.update({
            "execution_successful": False,
            "error_feedback": f"SQL 실행 실패: {error}. SQL 쿼리를 수정해주세요.",
            "status": "execution_failed"
        })
        return result

def verify_results(state: GraphState) -> GraphState:
    logger.info(f"verify_results 노드 실행 (재시도: {state.get('retry_count', 0)})")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터 분석 전문가입니다.
        SQL 쿼리 결과가 사용자의 원래 요청을 제대로 해결하는지 확인하는 것이 당신의 임무입니다.
        결과를 검증하고 명확하게 판단해주세요.
        응답 형식:
        VERIFIED: [이유] - 결과가 요청을 충족하는 경우
        NOT_VERIFIED: [이유] - 결과가 요청을 충족하지 않는 경우, 이유를 설명하고 개선 사항을 제안하세요."""),
        ("user", """사용자 요청: {user_request}
        
        SQL 쿼리: {sql}
        
        쿼리 결과: {results}
        
        이 결과가 사용자의 요청을 제대로 해결하나요? 'VERIFIED:' 또는 'NOT_VERIFIED:'로 시작하는 응답을 제공하세요.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "user_request": state["user_request"],
        "sql": state["sql"],
        "results": json.dumps(state["results"], default=str)
    })
    
    verification_text = response.content.strip()
    verification_passed = verification_text.startswith("VERIFIED:")
    
    logger.info(f"검증 결과: {'통과' if verification_passed else '실패'}")
    logger.info(f"검증 메시지: {verification_text[:200]}...")
    
    # 기본 상태 유지를 위한 딕셔너리
    result = {
        "sql": state["sql"],
        "user_request": state["user_request"],
        "results": state["results"],
        "retry_count": state.get("retry_count", 0),
        "is_valid": state.get("is_valid", False),
        "execution_successful": state.get("execution_successful", False)
    }
    
    if verification_passed:
        result.update({
            "verification_passed": True,
            "verification_message": verification_text,
            "status": "results_verified"
        })
        return result
    else:
        result.update({
            "verification_passed": False,
            "error_feedback": f"결과 검증 실패: {verification_text}. 더 나은 SQL 쿼리를 생성해주세요.",
            "status": "results_not_verified"
        })
        return result

def generate_insights(state: GraphState) -> GraphState:
    logger.info(f"generate_insights 노드 실행 (재시도: {state.get('retry_count', 0)})")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터에서 인사이트를 찾는 데이터 분석 전문가입니다.
        제공된 SQL 쿼리 결과를 분석하고 사용자 요청과 관련된 의미 있는 인사이트를 제공하세요.
        관련 통계, 추세, 이상치, 그래프 및 실행 가능한 권장 사항을 포함하세요.
        섹션과 글머리 기호를 사용하여 명확하고 구조화된 방식으로 응답을 포맷하세요.
         가능하다면 시각화 가능한 형태로 그래프를 뽑아주세요"""),
        ("user", """사용자 요청: {user_request}
        
        SQL 쿼리: {sql}
        
        쿼리 결과: {results}
        
        이 데이터를 기반으로 종합적인 인사이트를 제공하세요.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "user_request": state["user_request"],
        "sql": state["sql"],
        "results": json.dumps(state["results"], default=str)
    })
    
    logger.info("인사이트 생성 완료")
    
    # 이전 상태의 모든 정보 유지
    result = {
        **state,
        "insights": response.content,
        "status": "insights_generated"
    }
    
    return result

# 라우팅 함수
def route_validation(state: GraphState) -> str:
    is_valid = state.get("is_valid", False)
    logger.info(f"검증 라우팅: is_valid={is_valid}, 결정={('valid' if is_valid else 'invalid')}")
    
    # 검증 실패 시 SQL 로그
    if not is_valid:
        logger.info(f"검증 실패한 SQL: {state.get('sql', '없음')}")
        logger.info(f"오류 피드백: {state.get('error_feedback', '없음')}")
        logger.info(f"재시도 횟수: {state.get('retry_count', 0)}")
    
    # 최대 재시도 횟수 체크
    if not is_valid and state.get("retry_count", 0) >= 5:
        logger.warning("최대 재시도 횟수 도달, 검증 실패에도 불구하고 진행합니다.")
        return "valid"  # 강제 진행
    
    return "valid" if is_valid else "invalid"

def route_execution(state: GraphState) -> str:
    is_successful = state.get("execution_successful", False)
    logger.info(f"실행 라우팅: execution_successful={is_successful}, 결정={('success' if is_successful else 'failure')}")
    
    # 실행 실패 시 SQL 로그
    if not is_successful:
        logger.info(f"실행 실패한 SQL: {state.get('sql', '없음')}")
        logger.info(f"오류 피드백: {state.get('error_feedback', '없음')}")
        logger.info(f"재시도 횟수: {state.get('retry_count', 0)}")
    
    # 최대 재시도 횟수 체크
    if not is_successful and state.get("retry_count", 0) >= 5:
        logger.warning("최대 재시도 횟수 도달, 실행 실패에도 불구하고 진행합니다.")
        return "success"  # 강제 진행
    
    return "success" if is_successful else "failure"

def route_verification(state: GraphState) -> str:
    is_verified = state.get("verification_passed", False)
    logger.info(f"검증 라우팅: verification_passed={is_verified}, 결정={('verified' if is_verified else 'not_verified')}")
    
    # 검증 실패 시 상세 로그
    if not is_verified:
        logger.info(f"검증 실패한 SQL: {state.get('sql', '없음')}")
        logger.info(f"검증 메시지: {state.get('verification_message', '없음')}")
        logger.info(f"오류 피드백: {state.get('error_feedback', '없음')}")
        logger.info(f"재시도 횟수: {state.get('retry_count', 0)}")
    
    # 최대 재시도 횟수 체크
    if not is_verified and state.get("retry_count", 0) >= 5:
        logger.warning("최대 재시도 횟수 도달, 검증 실패에도 불구하고 진행합니다.")
        return "verified"  # 강제 진행
    
    return "verified" if is_verified else "not_verified"

# LangGraph 구축
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("verify_results", verify_results)
    workflow.add_node("generate_insights", generate_insights)
    
    # 시작점 설정
    workflow.set_entry_point("generate_sql")
    
    # 조건부 엣지 설정 - 모든 엣지에 merge=True 추가
    workflow.add_conditional_edges(
        "validate_sql",
        route_validation,
        {
            "valid": "execute_sql",
            "invalid": "generate_sql" 
        },
        merge=True
    )
    
    workflow.add_conditional_edges(
        "execute_sql",
        route_execution,
        {
            "success": "verify_results",
            "failure": "generate_sql"
        },
        merge=True
    )
    
    workflow.add_conditional_edges(
        "verify_results",
        route_verification,
        {
            "verified": "generate_insights",
            "not_verified": "generate_sql"
        },
        merge=True
    )
    
    # 종료점 설정
    workflow.add_edge("generate_insights", END)
    
    logger.info("그래프 컴파일됨")
    return workflow.compile()

# Streamlit UI
st.title("Redshift 자연어 쿼리 어시스턴트")

# 디버그 모드 설정 - 기본값을 True로 설정
debug_mode = st.sidebar.checkbox("디버그 모드", value=True)

# 개발 모드 설정 - 노드 직접 실행 여부
dev_mode = st.sidebar.checkbox("개발 모드 (노드 직접 실행)", value=True)

# 최대 재시도 횟수 설정
max_retries = st.sidebar.slider("최대 재시도 횟수", min_value=1, max_value=10, value=5)

# 타임아웃 설정
timeout = st.sidebar.slider("타임아웃(초)", min_value=30, max_value=600, value=180)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 입력
if prompt := st.chat_input("데이터에 대해 질문하세요..."):
    # 사용자 메시지를 채팅 기록에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 처리 중 표시
    with st.chat_message("assistant"):
        debug_container = st.empty()
        error_container = st.empty()
        progress_bar = st.progress(0)
        
        with st.spinner("요청을 분석 중입니다..."):
            try:
                # 초기 진행 상태 표시
                debug_info = {"상태": "시작됨", "단계": "그래프 빌드 중"}
                if debug_mode:
                    debug_container.json(debug_info)
                progress_bar.progress(10)
                
                # 결과 변수 초기화 - 오류 발생 시 기본값
                result = {"sql": "", "status": "failed", "retry_count": 0, "user_request": prompt}
                
                # 개발 모드에서는 노드를 직접 실행
                if dev_mode:
                    try:
                        # 초기 상태
                        current_state = {"user_request": prompt, "retry_count": 0}
                        
                        # SQL 생성 및 실행 프로세스
                        for attempt in range(max_retries):
                            # 1. SQL 생성
                            debug_info["단계"] = f"SQL 생성 중 (시도 {attempt+1}/{max_retries})"
                            if debug_mode:
                                debug_container.json(debug_info)
                            progress_bar.progress(20)
                            
                            sql_state = generate_sql(current_state)
                            
                            # 현재 결과 업데이트 - SQL 생성 결과
                            result = sql_state
                            
                            if debug_mode:
                                debug_container.json({**debug_info, "SQL 생성 결과": sql_state.get("sql", "")[:100] + "..."})
                            progress_bar.progress(30)
                            
                            # SQL 생성 결과 확인
                            if "sql" not in sql_state:
                                error_container.error("SQL 생성 실패: SQL이 생성되지 않았습니다.")
                                break  # SQL 생성 자체가 실패하면 종료
                            
                            # 2. SQL 검증
                            debug_info["단계"] = f"SQL 검증 중 (시도 {attempt+1}/{max_retries})"
                            if debug_mode:
                                debug_container.json(debug_info)
                            progress_bar.progress(40)
                            
                            # SQL 상태를 현재 상태에 복사
                            validate_state = {**current_state, **sql_state}
                            
                            # SQL 검증 함수 호출
                            validation_result = validate_sql_node(validate_state)
                            
                            # 현재 결과 업데이트 - 검증 결과
                            result = validation_result
                            
                            if debug_mode:
                                debug_container.json({**debug_info, "검증 결과": "성공" if validation_result.get("is_valid", False) else "실패"})
                            progress_bar.progress(50)
                            
                            # 검증 실패 시
                            if not validation_result.get("is_valid", False):
                                error_container.warning(f"SQL 검증 실패 (시도 {attempt+1}/{max_retries})")
                                if attempt < max_retries - 1:  # 아직 재시도 가능
                                    current_state = {
                                        "user_request": prompt,
                                        "error_feedback": validation_result.get("error_feedback", ""),
                                        "retry_count": attempt + 1
                                    }
                                    continue  # 다음 시도로
                                else:
                                    error_container.error("최대 시도 횟수 도달, 마지막 쿼리로 진행합니다.")
                                    # 강제로 다음 단계 진행
                                    validation_result["is_valid"] = True
                            
                            # 3. SQL 실행
                            debug_info["단계"] = f"SQL 실행 중 (시도 {attempt+1}/{max_retries})"
                            if debug_mode:
                                debug_container.json(debug_info)
                            progress_bar.progress(60)
                            
                            # 검증 상태와 SQL 상태 병합
                            execute_state = {**validate_state, **validation_result}
                            
                            # SQL 실행 함수 호출
                            execution_result = execute_sql_node(execute_state)
                            
                            # 현재 결과 업데이트 - 실행 결과
                            result = execution_result
                            
                            if debug_mode:
                                debug_container.json({**debug_info, "실행 결과": "성공" if execution_result.get("execution_successful", False) else "실패"})
                            progress_bar.progress(70)
                            
                            # 실행 실패 시
                            if not execution_result.get("execution_successful", False):
                                error_container.warning(f"SQL 실행 실패 (시도 {attempt+1}/{max_retries})")
                                if attempt < max_retries - 1:  # 아직 재시도 가능
                                    current_state = {
                                        "user_request": prompt,
                                        "error_feedback": execution_result.get("error_feedback", ""),
                                        "retry_count": attempt + 1
                                    }
                                    continue  # 다음 시도로
                                else:
                                    error_container.error("최대 시도 횟수 도달, 마지막 결과로 진행합니다.")
                                    break  # 루프 종료
                            
                            # 4. 결과 검증
                            debug_info["단계"] = f"결과 검증 중 (시도 {attempt+1}/{max_retries})"
                            if debug_mode:
                                debug_container.json(debug_info)
                            progress_bar.progress(80)
                            
                            # 실행 상태와 이전 상태 병합
                            verify_state = {**execute_state, **execution_result}
                            
                            # 결과 검증 함수 호출
                            verification_result = verify_results(verify_state)
                            
                            # 현재 결과 업데이트 - 검증 결과
                            result = verification_result
                            
                            if debug_mode:
                                debug_container.json({**debug_info, "검증 결과": "통과" if verification_result.get("verification_passed", False) else "실패"})
                            progress_bar.progress(85)
                            
                            # 검증
                            # 검증 실패 시
                            if not verification_result.get("verification_passed", False):
                                error_container.warning(f"결과 검증 실패 (시도 {attempt+1}/{max_retries})")
                                if attempt < max_retries - 1:  # 아직 재시도 가능
                                    current_state = {
                                        "user_request": prompt,
                                        "error_feedback": verification_result.get("error_feedback", ""),
                                        "retry_count": attempt + 1
                                    }
                                    continue  # 다음 시도로
                                else:
                                    error_container.error("최대 시도 횟수 도달, 마지막 결과로 진행합니다.")
                                    verification_result["verification_passed"] = True  # 강제 통과
                            
                            # 5. 인사이트 생성
                            debug_info["단계"] = "인사이트 생성 중"
                            if debug_mode:
                                debug_container.json(debug_info)
                            progress_bar.progress(90)
                            
                            # 검증 상태와 이전 상태 병합
                            insight_state = {**verify_state, **verification_result}
                            
                            # 인사이트 생성 함수 호출
                            insight_result = generate_insights(insight_state)
                            
                            # 현재 결과 업데이트 - 인사이트 결과
                            result = insight_result
                            
                            if debug_mode:
                                debug_container.json({**debug_info, "인사이트 생성 완료": True})
                            progress_bar.progress(100)
                            
                            # 성공적으로 완료되면 루프 종료
                            break
                        
                    except Exception as node_error:
                        error_container.error(f"노드 실행 중 오류 발생: {str(node_error)}")
                        import traceback
                        st.code(traceback.format_exc(), language="python")
                        
                        # 그래프 전체 실행으로 대체
                        debug_info["단계"] = "그래프 전체 실행으로 대체"
                        if debug_mode:
                            debug_container.json(debug_info)
                        dev_mode = False
                
                # 개발 모드가 아니거나, 노드 직접 실행에 실패한 경우 그래프 전체 실행
                if not dev_mode:
                    # 그래프 구축
                    graph = build_graph()
                    
                    # 타임아웃 및 재시도 설정
                    config = {
                        "recursion_limit": max_retries * 3,  # 재귀 제한 설정
                        "configurable": {
                            "timeout": timeout
                        }
                    }
                    
                    # 그래프 실행
                    debug_info["단계"] = "그래프 전체 실행 중"
                    if debug_mode:
                        debug_container.json(debug_info)
                    
                    result = graph.invoke({"user_request": prompt, "retry_count": 0}, config)
                    
                    if debug_mode:
                        debug_container.json({**debug_info, "실행 완료": True})
                    progress_bar.progress(100)
                
                # 결과 표시
                if debug_mode:
                    st.subheader("최종 상태")
                    st.json({
                        "status": result.get("status", "unknown"),
                        "verification_passed": result.get("verification_passed", False),
                        "execution_successful": result.get("execution_successful", False),
                        "is_valid": result.get("is_valid", False),
                        "retry_count": result.get("retry_count", 0),
                        "keys": list(result.keys())
                    })
                
                # SQL 표시
                if "sql" in result:
                    st.subheader("SQL 쿼리")
                    st.code(result["sql"], language="sql")
                else:
                    st.error("SQL 쿼리를 생성하지 못했습니다.")
                
                # 결과를 테이블로 표시 (가능한 경우)
                if "results" in result and result["results"]:
                    st.subheader("쿼리 결과")
                    df = pd.DataFrame(result["results"])
                    st.dataframe(df)
                
                # 인사이트 표시 (있는 경우)
                if "insights" in result and result["insights"]:
                    st.subheader("인사이트")
                    st.markdown(result["insights"])
                    
                    # 전체 응답을 채팅 기록에 저장
                    full_response = f"""**SQL 쿼리:**
```sql
{result.get("sql", "SQL 쿼리를 생성하지 못했습니다.")}
```

**인사이트:**
{result["insights"]}
"""
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    # 부분 결과라도 표시
                    st.warning("완전한 처리가 이루어지지 않았습니다. 부분 결과만 표시합니다.")
                    
                    # SQL이 생성된 경우에만 채팅 기록에 저장
                    if "sql" in result:
                        partial_response = f"""**SQL 쿼리:**
```sql
{result["sql"]}
```

**처리 상태:** 
SQL 쿼리는 생성되었지만, 완전한 분석이 이루어지지 않았습니다.
재시도 횟수: {result.get("retry_count", 0)}
"""
                        st.session_state.messages.append({"role": "assistant", "content": partial_response})
            
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")
                st.session_state.messages.append({"role": "assistant", "content": f"처리 중 오류가 발생했습니다: {str(e)}"})

# 사이드바에 정보 추가
with st.sidebar:
    st.header("소개")
    st.info("""이 애플리케이션은 자연어를 사용하여 Redshift 데이터를 쿼리할 수 있게 해줍니다.
    분석하고 싶은 내용을 설명하면, 시스템이 SQL을 생성하고 실행한 후,
    결과를 기반으로 인사이트를 제공합니다.""")
    
    # 도움말 추가
    st.header("문제 해결")
    st.warning("""
    - SQL 생성 후 멈추는 문제가 발생하면:
      1. '개발 모드'를 활성화하여 노드를 직접 실행해보세요.
      2. 디버그 모드를 통해 어느 단계에서 멈추는지 확인하세요.
      3. 타임아웃 값을 늘려 더 긴 처리 시간을 허용하세요.
    - 노드 간 상태 전달 문제가 주로 발생합니다. 각 노드 함수가 모든 필요한 키를 포함하는지 확인하세요.
    - 재시도 횟수를 조정하여 최대 몇 번까지 재시도할지 설정할 수 있습니다.
    - 최대 재시도 횟수에 도달하면 강제로 다음 단계로 진행합니다.
    """)