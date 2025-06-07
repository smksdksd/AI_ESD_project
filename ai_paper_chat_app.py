
import streamlit as st
import openai
import fitz  # PyMuPDF
import docx
import io

# API 키 설정 (Streamlit Cloud 배포 시 secrets.toml 사용 권장)
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "YOUR_API_KEY"

# 텍스트 추출 함수
def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_from_docx(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "지원되지 않는 파일 형식입니다."

def extract_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_from_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    return "\n".join([para.text for para in doc.paragraphs])

# GPT 요약 및 질의응답 함수
def ask_question_to_ai(context_text, user_question, chat_history=[]):
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
