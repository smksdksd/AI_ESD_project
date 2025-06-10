import streamlit as st
import openai

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ask_question_to_ai(context_text, user_question, chat_history=None):
    if chat_history is None:
        chat_history = []
    messages = [{"role": "system", "content": "너는 논문 요약 및 질의응답 AI야."}]
    messages += chat_history
    messages.append({"role": "user", "content": f"논문 내용:\n{context_text[:3000]}\n\n질문: {user_question}"})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=600,
        temperature=0.3
    )

    reply = response.choices[0].message.content
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": reply})
    return reply, chat_history

# 이하 Streamlit 코드 동일하게 사용