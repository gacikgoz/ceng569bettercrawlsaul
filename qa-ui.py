import streamlit as st
from irmodel import retrieve_and_rerank, answer_question, store, preprocess_query

# Initialize Streamlit
st.set_page_config(page_title="Financial Chat QA", layout="centered")
st.title("📊 Financial Question Answering Chatbot")

# Model Info
st.sidebar.title("🤖 Model Info")
st.sidebar.write("System: Hybrid IR + Reranker + QA")
st.sidebar.write("Retrieval: BM25 + Dense (FAISS)")
st.sidebar.write("Reranker: Cross-Encoder (MiniLM)")
st.sidebar.write("QA Model: deepset/roberta-base-squad2")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.clear()

# Chat State Init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if query := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # IR + Rerank
    try:
        qid = f"q_{len(st.session_state.messages)}"
        reranked = retrieve_and_rerank(qid, query)
        answers = answer_question(query, reranked)

        if not answers:
            answer_text = "❌ Sorry, I couldn’t find a good answer in the documents."
        else:
            answer_text = f"✅ **Answer:** {answers[0]['answer']}\n\n"
            answer_text += f"📌 *Snippet:* “{answers[0]['snippet']}”"

    except Exception as e:
        answer_text = f"⚠️ An error occurred: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    with st.chat_message("assistant"):
        st.markdown(answer_text)

