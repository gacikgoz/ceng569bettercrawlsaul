import streamlit as st

st.title("QA Demo UI")

def get_answer(question):
    return f"[Dummy answer] You asked: {question}"

question = st.text_input("Enter your question")

if st.button("Get Answer") and question:
    answer = get_answer(question)
    st.markdown(f"**Answer:** {answer}")
