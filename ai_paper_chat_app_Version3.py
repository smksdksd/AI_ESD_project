import streamlit as st
import openai
import fitz  # PyMuPDF
import docx
import io

# API 키 설정
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
elif "OPENAI_API_KEY" in st.session_state:
    openai.api_key = st.session_state["OPENAI_API_KEY"]
else:
    st.error("OpenAI API 키가 설정되지 않았습니다. secrets.toml이나 환경변수에 키를 추가하세요.")
    st.stop()

# 텍스트 추출 함수
def extract_text(file):
    file_bytes = file.read()  # 파일을 한 번만 읽음
    file_buffer = io.BytesIO(file_bytes)  # BytesIO로 감싸서 여러 번 사용 가능
    file_name = file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_from_pdf(file_buffer)
    elif file_name.endswith(".docx"):
        return extract_from_docx(file_buffer)
    elif file_name.endswith(".txt"):
        # 텍스트 파일은 원본 바이너리를 utf-8로 디코딩
        return file_bytes.decode("utf-8")
    else:
        return "지원되지 않는 파일 형식입니다."

def extract_from_pdf(file_buffer):
    file_buffer.seek(0)
    doc = fitz.open(stream=file_buffer.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_from_docx(file_buffer):
    file_buffer.seek(0)
    doc = docx.Document(file_buffer)
    return "\n".join([para.text for para in doc.paragraphs])

# GPT 요약 및 질의응답 함수
def ask_question_to_ai(context_text, user_question, chat_history=None):
    if chat_history is None:
        chat_history = []
    messages = [{"role": "system", "content": "너는 사용자의 논문을 요약하고, 질문에 응답하는 친절한 AI야."}]
    messages += chat_history
    messages.append({"role": "user", "content": f"논문 내용:\n{context_text[:3000]}\n\n질문: {user_question}"})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=600,
        temperature=0.3
    )

    reply = response.choices[0].message["content"]
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": reply})
    return reply, chat_history

# Streamlit UI
st.set_page_config(page_title="논문 요약 & QnA", page_icon="📄", layout="wide")
st.title("📄 AI 논문 요약 및 질의응답 서비스")
st.write("문서를 업로드하고 질문하면 AI가 요약하고 대답해줍니다.")

uploaded_file = st.file_uploader("논문 파일 업로드 (PDF, DOCX, TXT 지원)", type=["pdf", "docx", "txt"])
if uploaded_file:
    with st.spinner("문서 처리 중..."):
        doc_text = extract_text(uploaded_file)
        st.session_state["doc_text"] = doc_text
        st.success("문서 업로드 및 처리 완료!")

# 대화 기록 저장
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 채팅 입력
if "doc_text" in st.session_state:
    user_input = st.chat_input("논문에 대해 질문해보세요.")
    if user_input:
        with st.spinner("AI가 응답 중입니다..."):
            reply, updated_history = ask_question_to_ai(
                st.session_state["doc_text"], user_input, st.session_state["chat_history"]
            )
            st.session_state["chat_history"] = updated_history
            st.chat_message("user").write(user_input)
            st.chat_message("assistant").write(reply)