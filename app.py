import streamlit as st

st.set_page_config(page_title="Business Health Chatbot")

st.chat_message("test")
user_input = st.text_input("Let me help you, give me the core of yours problems in your buisness, and I will help you get a diagnosis and solutions")

if user_input:
    message = st.chat_message("human")
    message.write(f"You asked: {user_input}")

    answer = st.chat_message("ai")
    answer.write("Answer will appear here...")