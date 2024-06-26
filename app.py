import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import faiss

load_dotenv()

def main():
    st.header("💬 DocuChat - PDF Chat App")
    st.markdown(" DocuChat is an AI-powered research oriented chatbot application that allows users to upload PDF documents and interact with them by asking questions.")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF, ask questions and get answers.", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.index") and os.path.exists(f"{store_name}_chunks.pkl"):
            # Load FAISS index and chunks
            index = faiss.read_index(f"{store_name}.index")
            with open(f"{store_name}_chunks.pkl", "rb") as f:
                loaded_chunks = pickle.load(f)
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS(embedding=embeddings, index=index, texts=loaded_chunks)
            VectorStore.index = index
            st.write('Embeddings Loaded from Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            faiss.write_index(VectorStore.index, f"{store_name}.index")
            with open(f"{store_name}_chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)

        # Accept user questions/query
        query = st.text_input("Ask questions:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
    st.write('Built by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app/)')


if __name__ == '__main__':
    main()
