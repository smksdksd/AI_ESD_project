import streamlit as st
import os
import tempfile
import hashlib
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------
# 1. 페이지 설정 및 다국어 UI 텍스트 정의
# --------------------------------------------------------------------------

st.set_page_config(page_title="AI 논문 분석 Q&A", layout="wide")

LANGUAGES = {
    "한국어": "ko",
    "English": "en"
}
ANSWER_LANGUAGES = {
    "Auto/문서 기준": "auto",
    "한국어": "Korean",
    "English": "English"
}

def get_ui_labels(lang_code):
    if lang_code == "en":
        return {
            "title": "📄 AI Paper Analysis & Q&A",
            "upload_header": "1. Upload Paper",
            "file_uploader": "Select a file (PDF, DOCX, TXT)",
            "analyze_btn": "Start Analysis",
            "analyzing": "Analyzing the paper... This may take a moment.",
            "analyze_success": "Paper analysis completed! You can now ask questions.",
            "analyze_error": "Error during analysis:",
            "upload_first": "Please upload a file first.",
            "ask_header": "2. Ask Questions about the Paper",
            "ask_placeholder": "e.g., What are the main contributions of this paper?",
            "wait_answer": "Generating answer with ChatGPT...",
            "answer_header": "🤖 ChatGPT Answer",
            "answer_error": "Error during answer generation:",
            "need_upload": "Please upload a paper in the sidebar and click 'Start Analysis'.",
            "history_header": "Q&A History",
            "already_analyzed": "This paper has already been analyzed. Ask a new question or upload a different paper.",
            "current_paper": "Currently analyzing:"
        }
    else: # 한국어
        return {
            "title": "📄 AI 논문 분석 및 Q&A",
            "upload_header": "1. 논문 업로드",
            "file_uploader": "파일을 선택하세요 (PDF, DOCX, TXT)",
            "analyze_btn": "논문 분석 시작",
            "analyzing": "논문을 분석 중입니다. 잠시만 기다려주세요...",
            "analyze_success": "논문 분석이 완료되었습니다! 이제 질문을 할 수 있습니다.",
            "analyze_error": "분석 중 오류 발생:",
            "upload_first": "먼저 파일을 업로드해주세요.",
            "ask_header": "2. 논문에 대해 질문하기",
            "ask_placeholder": "예: 이 논문의 주요 기여는 무엇인가요?",
            "wait_answer": "ChatGPT가 답변을 생성 중입니다...",
            "answer_header": "🤖 ChatGPT 답변",
            "answer_error": "답변 생성 중 오류 발생:",
            "need_upload": "사이드바에서 논문을 업로드하고 '분석 시작' 버튼을 눌러주세요.",
            "history_header": "Q&A 기록",
            "already_analyzed": "이미 분석된 논문입니다. 새로운 질문을 하시거나 다른 논문을 업로드해주세요.",
            "current_paper": "현재 분석 중인 논문:"
        }

# --------------------------------------------------------------------------
# 2. 핵심 기능 함수 정의
# --------------------------------------------------------------------------

def get_file_hash(file_obj):
    file_bytes = file_obj.getvalue()
    return hashlib.md5(file_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def get_text_from_doc(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        pages = loader.load_and_split()
        text = " ".join(t.page_content for t in pages)
    finally:
        os.remove(tmp_file_path)
    return text

@st.cache_data(show_spinner=False)
def create_qa_chain(_text, answer_language):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(_text)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = FAISS.from_texts(chunks, embeddings)

    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """
    if answer_language != "auto":
        prompt_template += f"\nHelpful Answer (MUST be in {answer_language}):"
    else:
        prompt_template += "\nHelpful Answer:"

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --------------------------------------------------------------------------
# 3. Streamlit 세션 상태(Session State) 초기화 및 언어 변경 감지
# --------------------------------------------------------------------------

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = []
if "analyzed_filename" not in st.session_state:
    st.session_state.analyzed_filename = None
if "analyzed_filehash" not in st.session_state:
    st.session_state.analyzed_filehash = None
if "language" not in st.session_state:
    st.session_state.language = "ko"
if "answer_language" not in st.session_state:
    st.session_state.answer_language = "auto"
if "last_answer_language" not in st.session_state:
    st.session_state.last_answer_language = st.session_state.answer_language

# 답변 언어 변경 여부 감지
st.session_state.answer_language_changed = (
    st.session_state.last_answer_language != st.session_state.answer_language
)
st.session_state.last_answer_language = st.session_state.answer_language

# --------------------------------------------------------------------------
# 4. 사이드바 UI 구성
# --------------------------------------------------------------------------

with st.sidebar:
    selected_lang_key = st.selectbox(
        "🌐 언어 (Language/UI)",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.values()).index(st.session_state.language)
    )
    st.session_state.language = LANGUAGES[selected_lang_key]
    labels = get_ui_labels(st.session_state.language)

    selected_ans_lang_key = st.selectbox(
        "🤖 답변 언어 (Answer Language)",
        options=list(ANSWER_LANGUAGES.keys()),
        index=list(ANSWER_LANGUAGES.values()).index(st.session_state.answer_language)
    )
    st.session_state.answer_language = ANSWER_LANGUAGES[selected_ans_lang_key]

    st.markdown("---")
    st.header(labels["upload_header"])
    uploaded_file = st.file_uploader(
        labels["file_uploader"],
        type=['pdf', 'docx', 'txt']
    )

    if st.button(labels["analyze_btn"]):
        if uploaded_file is not None:
            current_file_hash = get_file_hash(uploaded_file)
            # 파일이 변경되었거나 답변 언어가 변경되었을 때만 새로 분석
            if (
                current_file_hash != st.session_state.analyzed_filehash
                or st.session_state.answer_language_changed
            ):
                with st.spinner(labels["analyzing"]):
                    try:
                        extracted_text = get_text_from_doc(uploaded_file)
                        st.session_state.qa_chain = create_qa_chain(
                            extracted_text, st.session_state.answer_language
                        )
                        st.session_state.analyzed_filename = uploaded_file.name
                        st.session_state.analyzed_filehash = current_file_hash
                        st.session_state.history = []
                        st.success(labels["analyze_success"])
                    except Exception as e:
                        st.error(f"{labels['analyze_error']} {e}")
            else:
                st.info(labels["already_analyzed"])
        else:
            st.warning(labels["upload_first"])

# --------------------------------------------------------------------------
# 5. 메인 화면 UI 구성
# --------------------------------------------------------------------------

st.title(labels["title"])
st.markdown("---")

if st.session_state.qa_chain is None:
    st.info(labels["need_upload"])
else:
    st.header(labels["ask_header"])
    file_display_name = st.session_state.analyzed_filename or "없음"
    st.markdown(f"**{labels['current_paper']}** `{file_display_name}`")

    query = st.text_input(
        labels["ask_placeholder"],
        key="query_input"
    )

    if query:
        with st.spinner(labels["wait_answer"]):
            try:
                response = st.session_state.qa_chain.invoke(query)
                answer = response['result']
                st.session_state.history.append({"question": query, "answer": answer})
                st.session_state.query_input = ""  # 입력창 자동 비우기
            except Exception as e:
                st.error(f"{labels['answer_error']} {e}")
                st.session_state.history.append({"question": query, "answer": f"Error: {e}"})
                st.session_state.query_input = ""

    # Q&A 기록 표시 (expander로 접기 기능)
    if st.session_state.history:
        st.markdown("---")
        st.subheader(labels["history_header"])
        for idx, qa in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Q{idx}: {qa['question']}"):
                st.markdown(f"**A:** {qa['answer']}")
