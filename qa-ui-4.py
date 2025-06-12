import streamlit as st

# Dummy model
class DummyModel:
    def __init__(self):
        self.name = "Dummy QA Model"
        self.version = "v0.1"

    def answer(self, question):
        if "capital" in question.lower():
            return "The capital is Paris."
        return f"I don't know the answer to: '{question}'"

# Başlangıç
st.set_page_config(page_title="Chat QA", layout="centered")
st.title("🧠 Question Answering Chatbot")

# Dummy model başlat
model = DummyModel()

# İlk kez çalışıyorsa mesaj geçmişi oluştur
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini göster
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Yeni giriş
if question := st.chat_input("Ask me anything..."):
    # Kullanıcının mesajı
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Model cevabı
    answer = model.answer(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Alt bilgi
st.sidebar.title("🤖 Model Info")
st.sidebar.write(f"Model: {model.name}")
st.sidebar.write(f"Version: {model.version}")
st.sidebar.button("🗑️ Clear Chat", on_click=lambda: st.session_state.clear())
