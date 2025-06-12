import streamlit as st

# Dummy model sınıfı
class DummyModel:
    def __init__(self):
        self.name = "Dummy QA Model"
        self.version = "v0.1"
    
    def answer(self, question):
        # Basit dummy cevaplama mantığı
        if "capital" in question.lower():
            return "The capital is Paris."
        elif "who wrote" in question.lower():
            return "Harper Lee wrote 'To Kill a Mockingbird'."
        elif "relativity" in question.lower():
            return "Relativity is a theory by Einstein explaining space-time."
        else:
            return f"I don't know the answer to: '{question}'"

# Bu UI'da dummy model kullanılıyor
def main():
    st.title("Question Answering UI")
    st.write("Ask a question and get an answer from the model.")

    # Dummy model başlatılıyor
    model = DummyModel()
    st.success("Model initialized successfully.")
    st.info("You can ask any question related to the model's knowledge base.")
    st.markdown("Type your question below and click **'Get Answer'**.")
    st.warning("Note: The model is designed to answer questions based on dummy logic for UI testing.")

    st.subheader("Instructions:")
    st.markdown("""
    1. Enter your question in the input box below.  
    2. Click the **'Get Answer'** button to receive an answer.  
    3. You can ask multiple questions by repeating the process.  
    """)

    st.subheader("Example Questions:")
    st.write("- What is the capital of France?")
    st.write("- Who wrote 'To Kill a Mockingbird'?")
    st.write("- Explain the theory of relativity.")

    question = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        if question:
            answer = model.answer(question)
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question to get an answer.")

    st.subheader("Model Information:")
    st.write(f"Model Name: {model.name}")
    st.write(f"Model Version: {model.version}")

if __name__ == "__main__":
    main()
