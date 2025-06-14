import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 함수 정의 ---

# 업로드된 파일에서 텍스트를 추출하는 함수
def get_text_from_doc(doc_file):
    # Streamlit의 UploadedFile 객체를 처리하기 위해 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(doc_file.getvalue())
        tmp_file_path = tmp_file.name

    # PyPDFLoader를 사용하여 PDF 파일 로드
    # (docx, txt 등 다른 형식 지원을 원하시면 여기에 로직을 추가하세요)
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load_and_split()
    text = " ".join(t.page_content for t in pages)
    
    # 임시 파일 삭제
    os.remove(tmp_file_path)
    return text

# 텍스트를 받아 QA 체인을 생성하고 반환하는 함수
def create_qa_chain(text):
    # 1. 텍스트를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # 청크 크기를 조금 더 크게 설정
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 2. OpenAI 임베딩 모델로 텍스트 청크를 벡터로 변환
    # st.secrets를 통해 안전하게 API 키를 가져옵니다.
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # 3. FAISS 벡터 스토어에 벡터화된 텍스트 저장
    vector_store = FAISS.from_texts(chunks, embeddings)

    # 4. ChatGPT 모델 지정 (gpt-4o 또는 gpt-3.5-turbo 등)
    llm = ChatOpenAI(
        model_name="gpt-4o", 
        temperature=0.7, # 창의적인 답변이 필요하면 온도를 높임
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    # 5. Langchain의 RetrievalQA 체인 생성
    # retriever는 벡터 스토어에서 관련 문서를 찾는 역할을 합니다.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 가장 일반적인 체인 타입
        retriever=vector_store.as_retriever()
    )
    return qa_chain

# --- Streamlit UI 구성 ---

st.set_page_config(page_title="AI 논문 분석 Q&A", layout="wide")
st.title("📄 AI 논문 분석 및 Q&A")
st.markdown("---")

# 세션 상태(session_state) 초기화
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# 사이드바 UI
with st.sidebar:
    st.header("1. 논문 업로드")
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요.", 
        type=['pdf']
    )
    
    if st.button("논문 분석 시작"):
        if uploaded_file is not None:
            with st.spinner('논문을 분석 중입니다. 텍스트 추출, 분할, 임베딩 과정이 진행됩니다...'):
                try:
                    extracted_text = get_text_from_doc(uploaded_file)
                    st.session_state.qa_chain = create_qa_chain(extracted_text)
                    st.success("논문 분석이 완료되었습니다! 이제 질문을 할 수 있습니다.")
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")
        else:
            st.warning("먼저 PDF 파일을 업로드해주세요.")

# 메인 화면 UI
st.header("2. 논문에 대해 질문하기")

if st.session_state.qa_chain is None:
    st.info("사이드바에서 논문을 업로드하고 '분석 시작' 버튼을 눌러주세요.")
else:
    query = st.text_input(
        "질문을 입력하세요:", 
        placeholder="예: 이 논문의 주요 기여는 무엇인가요?",
        key="query_input"
    )
    
    if query:
        with st.spinner("ChatGPT가 답변을 생성 중입니다..."):
            try:
                response = st.session_state.qa_chain.invoke(query)
                st.markdown("### 🤖 ChatGPT 답변")
                st.write(response['result'])
            except Exception as e:
                st.error(f"답변 생성 중 오류 발생: {e}")
