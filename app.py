import streamlit as st
import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import psycopg2
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List, Dict
import json
import os
from dotenv import load_dotenv

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
        return None

# Redshift에서 테이블 스키마 가져오기
def get_table_schema():
    conn = get_redshift_connection()
    if not conn:
        return "Redshift 연결에 실패했습니다."
    
    cursor = conn.cursor()
    try:
        # 스키마 정보를 가져오는 쿼리
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
        
        # 스키마 정보를 텍스트로 포맷팅
        schema_text = ""
        for schema in schema_info:
            schema_text += f"스키마: {schema}\n"
            for table in schema_info[schema]:
                schema_text += f"  테이블: {table}\n"
                for col in schema_info[schema][table]:
                    schema_text += f"    컬럼: {col['column']}, 타입: {col['data_type']}\n"
        
        return schema_text
    except Exception as e:
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
        # EXPLAIN을 사용하여 SQL 실행 없이 검증
        cursor.execute(f"EXPLAIN {sql}")
        return True, "SQL이 유효합니다."
    except Exception as e:
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
        return None, str(e)
    finally:
        cursor.close()
        conn.close()

# LangGraph 노드
def generate_sql(state):
    st.write("🔄 SQL 생성 중...")
    
    # 재시도 횟수 추적
    retry_count = state.get("retry_count", 0)
    if retry_count >= 3:
        st.write("⚠️ 최대 재시도 횟수 초과. 프로세스를 중단합니다.")
        return {"sql": "-- 최대 재시도 횟수 초과", "error_feedback": "최대 재시도 횟수를 초과했습니다."}
    
    schema = get_table_schema()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 Amazon Redshift SQL 전문가입니다.
        자연어 쿼리를 정확한 SQL 쿼리로 변환하는 것이 당신의 임무입니다.
        제공된 스키마 정보를 사용하여 정확한 SQL을 작성하세요.
        설명이나 마크다운 형식 없이 SQL 쿼리만 반환하세요.
        Redshift에 최적화된 SQL을 작성하세요."""),
        ("user", """스키마 정보:
        {schema}
        
        사용자 요청: {user_request}
        
        {error_feedback}
        
        이 요청에 대한 SQL 쿼리를 생성하세요.""")
    ])
    
    error_feedback = state.get("error_feedback", "")
    if error_feedback:
        st.write(f"⚠️ 이전 오류: {error_feedback}")
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "schema": schema,
            "user_request": state["user_request"],
            "error_feedback": error_feedback
        })
        
        sql = response.content.strip()
        # 마크다운 형식이 있으면 제거
        if sql.startswith("```sql"):
            sql = sql.replace("```sql", "").replace("```", "").strip()
        
        st.write("✅ SQL 생성 완료")
        st.code(sql, language="sql")
        return {"sql": sql, "retry_count": retry_count}
    except Exception as e:
        st.write(f"❌ SQL 생성 오류: {str(e)}")
        return {"sql": "", "error_feedback": f"SQL 생성 중 오류 발생: {str(e)}", "retry_count": retry_count + 1}

def validate_sql_node(state):
    st.write("🔄 SQL 검증 중...")
    sql = state["sql"]
    
    # 재시도 횟수 가져오기
    retry_count = state.get("retry_count", 0)
    
    is_valid, error_message = validate_sql(sql)
    
    if is_valid:
        st.write("✅ SQL 검증 성공")
        return {"is_valid": True, "retry_count": retry_count}
    else:
        st.write(f"❌ SQL 검증 실패: {error_message}")
        return {
            "is_valid": False,
            "error_feedback": f"SQL 검증 실패: {error_message}. SQL 쿼리를 수정해주세요.",
            "retry_count": retry_count + 1
        }

def execute_sql_node(state):
    st.write("🔄 SQL 실행 중...")
    sql = state["sql"]
    
    # 재시도 횟수 가져오기
    retry_count = state.get("retry_count", 0)
    
    df, error = execute_sql(sql)
    
    if df is not None:
        # DataFrame을 JSON 직렬화를 위해 딕셔너리로 변환
        results = df.to_dict(orient="records")
        st.write(f"✅ SQL 실행 성공: {len(results)}개의 결과 반환")
        return {"results": results, "execution_successful": True, "retry_count": retry_count}
    else:
        st.write(f"❌ SQL 실행 실패: {error}")
        return {
            "execution_successful": False,
            "error_feedback": f"SQL 실행 실패: {error}. SQL 쿼리를 수정해주세요.",
            "retry_count": retry_count + 1
        }

def verify_results(state):
    st.write("🔄 결과 검증 중...")
    
    # 재시도 횟수 가져오기
    retry_count = state.get("retry_count", 0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터 분석 전문가입니다.
        SQL 쿼리 결과가 사용자의 원래 요청을 제대로 해결하는지 확인하는 것이 당신의 임무입니다.
        결과가 요청과 일치하지 않는 경우, 이유를 설명하고 개선 사항을 제안하세요."""),
        ("user", """사용자 요청: {user_request}
        
        SQL 쿼리: {sql}
        
        쿼리 결과: {results}
        
        이 결과가 사용자의 요청을 제대로 해결하나요? 그렇지 않다면, 이유를 설명하세요.""")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "user_request": state["user_request"],
            "sql": state["sql"],
            "results": json.dumps(state["results"], default=str)
        })
        
        # 검증이 결과가 부적절하다고 제안하는지 확인
        verification_text = response.content.lower()
        verification_passed = "예" in verification_text[:100] and "아니오" not in verification_text[:100]
        
        if verification_passed:
            st.write("✅ 결과 검증 성공")
            return {"verification_passed": True, "verification_message": response.content, "retry_count": retry_count}
        else:
            st.write(f"❌ 결과 검증 실패: {response.content[:100]}...")
            return {
                "verification_passed": False,
                "error_feedback": f"결과 검증 실패: {response.content}. 더 나은 SQL 쿼리를 생성해주세요.",
                "retry_count": retry_count + 1
            }
    except Exception as e:
        st.write(f"❌ 결과 검증 오류: {str(e)}")
        return {"verification_passed": False, "error_feedback": f"결과 검증 중 오류 발생: {str(e)}", "retry_count": retry_count + 1}

def generate_insights(state):
    st.write("🔄 인사이트 생성 중...")
    
    # 재시도 횟수 가져오기
    retry_count = state.get("retry_count", 0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터에서 인사이트를 찾는 데이터 분석 전문가입니다.
        제공된 SQL 쿼리 결과를 분석하고 사용자 요청과 관련된 의미 있는 인사이트를 제공하세요.
        관련 통계, 추세, 이상치 및 실행 가능한 권장 사항을 포함하세요.
        섹션과 글머리 기호를 사용하여 명확하고 구조화된 방식으로 응답을 포맷하세요."""),
        ("user", """사용자 요청: {user_request}
        
        SQL 쿼리: {sql}
        
        쿼리 결과: {results}
        
        이 데이터를 기반으로 종합적인 인사이트를 제공하세요.""")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "user_request": state["user_request"],
            "sql": state["sql"],
            "results": json.dumps(state["results"], default=str)
        })
        
        st.write("✅ 인사이트 생성 완료")
        return {"insights": response.content, "retry_count": retry_count}
    except Exception as e:
        st.write(f"❌ 인사이트 생성 오류: {str(e)}")
        return {"insights": f"인사이트 생성 중 오류가 발생했습니다: {str(e)}", "retry_count": retry_count}

# LangGraph 구축
def build_graph():
    # 상태 정의
    class GraphState(TypedDict):
        user_request: str
        sql: Optional[str]
        is_valid: Optional[bool]
        results: Optional[List[Dict]]
        execution_successful: Optional[bool]
        verification_passed: Optional[bool]
        insights: Optional[str]
        error_feedback: Optional[str]
        retry_count: Optional[int]  # 재시도 횟수 추적
    
    # 그래프 생성
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("verify_results", verify_results)
    workflow.add_node("generate_insights", generate_insights)
    
    # 시작점 추가
    workflow.add_edge(START, "generate_sql")
    
    # 기본 플로우 정의
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_edge("validate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "verify_results")
    workflow.add_edge("verify_results", "generate_insights")
    workflow.add_edge("generate_insights", END)
    
    # 오류 처리를 위한 조건부 에지 추가
    workflow.add_conditional_edges(
        "validate_sql",
        lambda x: "is_valid" if x["is_valid"] else "not_valid",
        {
            "is_valid": "execute_sql",
            "not_valid": "generate_sql"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_sql",
        lambda x: "success" if x["execution_successful"] else "failure",
        {
            "success": "verify_results",
            "failure": "generate_sql"
        }
    )
    
    workflow.add_conditional_edges(
        "verify_results",
        lambda x: "passed" if x["verification_passed"] else "failed",
        {
            "passed": "generate_insights",
            "failed": "generate_sql"
        }
    )
    
    return workflow.compile()

# Streamlit UI
st.title("Redshift 자연어 쿼리 어시스턴트")

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
        with st.spinner("요청을 분석 중입니다..."):
            try:
                # 그래프 구축 및 실행
                st.write("🚀 워크플로우 시작")
                graph = build_graph()
                result = graph.invoke({
                    "user_request": prompt,
                    "retry_count": 0  # 초기 재시도 횟수 설정
                })
                st.write("✅ 워크플로우 완료")
                
                # SQL 표시
                st.subheader("SQL 쿼리")
                st.code(result["sql"], language="sql")
                
                # 결과를 테이블로 표시 (가능한 경우)
                if "results" in result and result["results"]:
                    st.subheader("쿼리 결과")
                    df = pd.DataFrame(result["results"])
                    st.dataframe(df)
                
                # 인사이트 표시
                st.subheader("인사이트")
                st.markdown(result["insights"])
                
                # 전체 응답을 채팅 기록에 저장
                full_response = f"""**SQL 쿼리:**
```sql
{result["sql"]}
```

**인사이트:**
{result["insights"]}
"""
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # 디버그 정보 저장
                st.session_state.current_state = result
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                st.write("디버그 정보:")
                st.write(f"오류 유형: {type(e).__name__}")
                st.write(f"오류 메시지: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")

# 사이드바에 정보 추가
with st.sidebar:
    st.header("소개")
    st.info("""이 애플리케이션은 자연어를 사용하여 Redshift 데이터를 쿼리할 수 있게 해줍니다.
    분석하고 싶은 내용을 설명하면, 시스템이 SQL을 생성하고 실행한 후,
    결과를 기반으로 인사이트를 제공합니다.""")
    
    st.header("디버그 모드")
    if st.checkbox("디버그 모드 활성화"):
        st.write("현재 상태:")
        if "current_state" in st.session_state:
            st.json(st.session_state.current_state)
