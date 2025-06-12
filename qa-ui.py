import streamlit as st
from irmodel import answer_question

# This is a simple question-answering UI using the irmodel library.
def main():
    st.title("Question Answering UI")
    st.write("Ask a question and get an answer from the model.")

    # Initialize the model
    model = answer_question()
    st.write("Model initialized successfully.")
    st.write("You can ask any question related to the model's knowledge base.")
    st.write("Type your question below and click 'Get Answer'.")
    st.write("Note: The model is designed to answer questions based on its training data.")
    st.write("Make sure to ask clear and concise questions for the best results.")
    # Display instructions
    st.write("### Instructions:")
    st.write("1. Enter your question in the input box below.")
    st.write("2. Click the 'Get Answer' button to receive an answer from the model.")
    st.write("3. The model will provide an answer based on its training data.")
    st.write("4. If the model cannot answer, it will inform you accordingly.")
    st.write("5. You can ask multiple questions by entering a new question and clicking 'Get Answer' again.")
    st.write("6. The model version is displayed below for reference.")
    st.write("### Example Questions:")
    st.write("- What is the capital of France?")
    st.write("- Who wrote 'To Kill a Mockingbird'?")
    st.write("- Explain the theory of relativity.")
    st.write("### Note:")
    st.write("The model's knowledge is based on its training data, which may not include the most recent information. For the latest updates, refer to official sources.")
    st.write("### Model Information:") 
    # Input for the question
    question = st.text_input("Enter your question here:")
    if st.button("Get Answer"):
        if question:
            answer = model.answer(question)
            st.write("### Answer:")
            st.write(answer)
        else:
            st.write("Please enter a question to get an answer.")
    # Display model information
    st.write("### Model Information:")
    st.write(f"Model Name: {model.name}") 
    st.write(f"Model Version: {model.version}")
if __name__ == "__main__":
    main()
# qa-ui.py
# This script creates a simple Streamlit application for question answering.
# It uses the irmodel library to handle the question-answering logic.
# To run this application, save it as qa-ui.py and run the command:
# streamlit run qa-ui.py