import streamlit as st
from irmodel import retrieve_and_rerank, answer_question

st.set_page_config(page_title="IR + QA Demo", layout="wide")
st.title("🔍 Information Retrieval + Question Answering")

query = st.text_input("Enter your question:", "")

if query:
    with st.spinner("Retrieving and answering..."):
        # ad-hoc bir query_id veriyoruz
        ranked = retrieve_and_rerank("q1", query)
        answers = answer_question(query, ranked)

    st.subheader("📄 Top Answers")
    for ans in answers:
        st.markdown(f"**Answer:** {ans['answer']}")
        st.markdown(f"_Score:_ {ans['score']:.3f}")
        st.markdown(f"> {ans['snippet']}")
        st.markdown("---")

    st.success("Done!")
else:   
    st.info("Please enter a question to get started.")