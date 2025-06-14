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
    "English": "en",
    "日本語": "ja",
    "中文": "zh"
}
ANSWER_LANGUAGES = {
    "Auto/문서 기준": "auto",
    "한국어": "Korean",
    "English": "English",
    "日本語": "Japanese",
    "中文": "Chinese"
}

def get_ui_labels(lang_code):
    if lang_code == "en":
        return {
            "title": "📄 AI Paper Analysis & Q&A",
            "upload_header": "1. Settings & Upload",
            "file_uploader": "Select a file (PDF, DOCX, TXT)",
            "analyze_btn": "Start Analysis",
            "analyzing": "Analyzing the paper... This may take a moment.",
            "analyze_success": "Analysis complete! You can now ask questions.",
            "analyze_error": "Error during analysis:",
            "upload_first": "Please upload a file first.",
            "ask_header": "2. Ask Questions",
            "ask_placeholder": "e.g., What are the main contributions of this paper?",
            "wait_answer": "Generating answer with ChatGPT...",
            "answer_header": "🤖 ChatGPT Answer",
            "answer_error": "Error during answer generation:",
            "need_upload": "Please upload and analyze a paper first.",
            "history_header": "Q&A History",
            "already_analyzed": "This paper has been analyzed.",
            "current_paper": "Currently analyzing:",
            "ui_lang_label": "🌐 UI Language",
            "ans_lang_label": "🤖 Answer Language"
        }
    elif lang_code == "ja":
        return {
            "title": "📄 AI論文分析 & Q&A",
            "upload_header": "1. 設定とアップロード",
            "file_uploader": "ファイルを選択してください (PDF, DOCX, TXT)",
            "analyze_btn": "解析開始",
            "analyzing": "論文を解析中です...しばらくお待ちください。",
            "analyze_success": "解析が完了しました！質問できます。",
            "analyze_error": "解析中にエラーが発生しました：",
            "upload_first": "まずファイルをアップロードしてください。",
            "ask_header": "2. 質問する",
            "ask_placeholder": "例：この論文の主な貢献は何ですか？",
            "wait_answer": "ChatGPTが回答を生成しています...",
            "answer_header": "🤖 ChatGPTの回答",
            "answer_error": "回答生成中にエラーが発生しました：",
            "need_upload": "まず論文をアップロードして解析してください。",
            "history_header": "Q&A履歴",
            "already_analyzed": "この論文はすでに解析されています。",
            "current_paper": "現在解析中の論文：",
            "ui_lang_label": "🌐 UI言語",
            "ans_lang_label": "🤖 回答言語"
        }
    elif lang_code == "zh":
        return {
            "title": "📄 AI论文分析与问答",
            "upload_header": "1. 设置与上传",
            "file_uploader": "请选择文件 (PDF, DOCX, TXT)",
            "analyze_btn": "开始分析",
            "analyzing": "正在分析论文……请稍候。",
            "analyze_success": "分析完成！现在可以提问。",
            "analyze_error": "分析时出错：",
            "upload_first": "请先上传文件。",
            "ask_header": "2. 提问",
            "ask_placeholder": "例如：这篇论文的主要贡献是什么？",
            "wait_answer": "ChatGPT正在生成答案……",
            "answer_header": "🤖 ChatGPT答案",
            "answer_error": "生成答案时出错：",
            "need_upload": "请先上传并分析论文。",
            "history_header": "Q&A记录",
            "already_analyzed": "该论文已被分析。",
            "current_paper": "当前分析的论文：",
            "ui_lang_label": "🌐 UI语言",
            "ans_lang_label": "🤖 答案语言"
        }
    else:  # "ko"
        return {
            "title": "📄 AI 논문 분석 및 Q&A",
            "upload_header": "1. 설정 및 업로드",
            "file_uploader": "파일을 선택하세요 (PDF, DOCX, TXT)",
            "analyze_btn": "논문 분석 시작",
            "analyzing": "논문을 분석 중입니다. 잠시만 기다려주세요...",
            "analyze_success": "분석이 완료되었습니다! 이제 질문할 수 있습니다.",
            "analyze_error": "분석 중 오류 발생:",
            "upload_first": "먼저 파일을 업로드해주세요.",
            "ask_header": "2. 논문에 대해 질문하기",
            "ask_placeholder": "예: 이 논문의 주요 기여는 무엇인가요?",
            "wait_answer": "ChatGPT가 답변을 생성 중입니다...",
            "answer_header": "🤖 ChatGPT 답변",
            "answer_error": "답변 생성 중 오류 발생:",
            "need_upload": "먼저 논문을 업로드하고 분석해주세요.",
            "history_header": "Q&A 기록",
            "already_analyzed": "이미 분석된 논문입니다.",
            "current_paper": "현재 분석 중인 논문:",
            "ui_lang_label": "🌐 UI 언어",
            "ans_lang_label": "🤖 답변 언어"
        }

# --------------------------------------------------------------------------
# 이하 코드는 기존과 동일 (생략)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 2. 핵심 기능 함수 정의 (캐싱 적용)
# --------------------------------------------------------------------------

def get_file_hash(file_obj):
    """파일의 내용으로 MD5 해시를 생성하여 반환합니다."""
    file_bytes = file_obj.getvalue()
    return hashlib.md5(file_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def get_text_from_doc(file_bytes, filename):
    """업로드된 파일 바이트에서 텍스트를 추출합니다."""
    suffix = os.path.splitext(filename)[-1].lower()
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

@st.cache_resource(show_spinner=False)
def create_qa_chain(_text, answer_language):
    """텍스트와 답변 언어를 받아 LangChain QA 체인을 생성하고 캐싱합니다."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(_text)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = FAISS.from_texts(chunks, embeddings)
    prompt_template = (
        "Use the following pieces of context to answer the question at the end.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
        "{context}\n\nQuestion: {question}\n"
    )
    if answer_language != "auto":
        prompt_template += f"\nHelpful Answer (MUST be in {answer_language}):"
    else:
        prompt_template += "\nHelpful Answer:"
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=st.secrets["OPENAI_API_KEY"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

# --------------------------------------------------------------------------
# 3. Streamlit 세션 상태(Session State) 초기화
# --------------------------------------------------------------------------

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = []
if "analyzed_filehash" not in st.session_state:
    st.session_state.analyzed_filehash = None
if "analyzed_filename" not in st.session_state:
    st.session_state.analyzed_filename = None
if "language" not in st.session_state:
    st.session_state.language = "ko"
if "answer_language" not in st.session_state:
    st.session_state.answer_language = "auto"
if "last_answer_language" not in st.session_state:
    st.session_state.last_answer_language = st.session_state.answer_language

# --------------------------------------------------------------------------
# 4. 메인 화면 UI 구성
# --------------------------------------------------------------------------

labels = get_ui_labels(st.session_state.language)
st.title(labels["title"])
st.markdown("---")

st.header(labels["upload_header"])
col1, col2 = st.columns(2)
with col1:
    selected_lang_key = st.selectbox(
        labels["ui_lang_label"],
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.values()).index(st.session_state.language)
    )
    st.session_state.language = LANGUAGES[selected_lang_key]
with col2:
    selected_ans_lang_key = st.selectbox(
        labels["ans_lang_label"],
        options=list(ANSWER_LANGUAGES.keys()),
        index=list(ANSWER_LANGUAGES.values()).index(st.session_state.answer_language)
    )
    st.session_state.answer_language = ANSWER_LANGUAGES[selected_ans_lang_key]

# UI 언어 변경 시 라벨 즉시 업데이트
labels = get_ui_labels(st.session_state.language)

uploaded_file = st.file_uploader(
    labels["file_uploader"],
    type=['pdf', 'docx', 'txt'],
    help="최대 20MB 파일 권장 (용량이 클수록 분석이 오래 걸릴 수 있습니다)"
)
analyze_button = st.button(labels["analyze_btn"], type="primary", use_container_width=True)

answer_language_changed = st.session_state.last_answer_language != st.session_state.answer_language

if analyze_button:
    if uploaded_file is not None:
        current_file_hash = get_file_hash(uploaded_file)
        if current_file_hash != st.session_state.analyzed_filehash or answer_language_changed:
            with st.spinner(labels["analyzing"]):
                try:
                    file_bytes = uploaded_file.getvalue()
                    extracted_text = get_text_from_doc(file_bytes, uploaded_file.name)
                    st.session_state.qa_chain = create_qa_chain(extracted_text, st.session_state.answer_language)
                    st.session_state.analyzed_filename = uploaded_file.name
                    st.session_state.analyzed_filehash = current_file_hash
                    st.session_state.history = []
                    st.session_state.last_answer_language = st.session_state.answer_language
                    st.success(labels["analyze_success"])
                    st.rerun()
                except Exception as e:
                    st.error(f"{labels['analyze_error']} {str(e)[:300]}")
        else:
            st.info(labels["already_analyzed"])
    else:
        st.warning(labels["upload_first"])

st.markdown("---")

# --------------------------------------------------------------------------
# 5. Q&A 및 기록 표시 섹션
# --------------------------------------------------------------------------

if st.session_state.qa_chain:
    st.header(labels["ask_header"])
    # 파일명이 너무 길면 중간 생략
    max_filename_len = 36
    show_filename = (
        st.session_state.analyzed_filename if st.session_state.analyzed_filename and len(st.session_state.analyzed_filename) <= max_filename_len
        else (st.session_state.analyzed_filename[:16] + "..." + st.session_state.analyzed_filename[-16:]) if st.session_state.analyzed_filename else "없음"
    )
    st.markdown(f"**{labels['current_paper']}** `{show_filename}`")

    with st.form(key="question_form", clear_on_submit=True):
        query = st.text_input(
            "질문:",
            placeholder=labels["ask_placeholder"],
            label_visibility="collapsed"
            # autofocus=True  # TypeError 방지를 위해 옵션 제거
        )
        submitted = st.form_submit_button("질문하기")

    if submitted and query:
        with st.spinner(labels["wait_answer"]):
            try:
                response = st.session_state.qa_chain.invoke(query)
                st.session_state.history.append({"question": query, "answer": response['result']})
            except Exception as e:
                st.error(f"{labels['answer_error']} {str(e)[:300]}")
                st.session_state.history.append({"question": query, "answer": f"Error: {e}"})

    if st.session_state.history:
        st.subheader(labels["history_header"])
        MAX_Q_LEN = 60
        total = len(st.session_state.history)
        for i, qa in enumerate(reversed(st.session_state.history)):
            short_q = qa['question'] if len(qa['question']) <= MAX_Q_LEN else qa['question'][:MAX_Q_LEN] + "..."
            q_number = total - i
            with st.expander(f"Q{q_number}: {short_q}"):
                st.markdown(f"**A:** {qa['answer']}")
        # Q&A 다운로드 버튼
        st.download_button(
            "Q&A 기록 다운로드",
            data="\n\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in st.session_state.history]),
            file_name="qa_history.txt"
        )
else:
    st.info(labels["need_upload"])
