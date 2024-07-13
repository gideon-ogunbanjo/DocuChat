import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not available in the provided context, 
        respond with "I'm sorry, the answer is not available in the context you have provided". Do not provide incorrect answers.\n\n
        Context: {context}?\n
        Question: {question}\n

        Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    return response['output_text']

def main():
    st.set_page_config(
        page_icon="💬",
        page_title="DocuChat",
        layout="centered"
    )
    st.header("💬 DocuChat - PDF Research Chat Tool")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_files = st.file_uploader("Upload your PDF, click on the submit button, ask questions and get answers.", accept_multiple_files=True)

    if uploaded_files:
        pdf_docs = [file for file in uploaded_files if file.type == "application/pdf"]
        non_pdf_files = [file.name for file in uploaded_files if file.type != "application/pdf"]
        if non_pdf_files:
            st.warning(f"Please upload PDF files only. The '{', '.join(non_pdf_files)}' file(s) are not PDFs.")

    if st.button("Submit & Process"):
        if not uploaded_files:
            st.warning("Please upload a file first.")
        elif not pdf_docs:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("File Processed Successfully")

    # Existing chat messages
    for message in st.session_state.messages:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])

    # User input text box
    if prompt := st.chat_input("Ask a question about your PDF files:"):

        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Typing animation
        with st.spinner("Typing..."):
            response_content = user_input(prompt)

        # Assistant message
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        with st.chat_message("assistant"):
            st.markdown(response_content)
            
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    # Footer with link
    link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
    st.markdown(link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
