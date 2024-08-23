import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

embeddings = OpenAIEmbeddings(
    model=st.secrets["EMBEDDINGS_MODEL_NAME"],
    openai_api_key=st.secrets["OPENAI_API_KEY"],
)

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model_name=st.secrets["LLM_MODEL_NAME"],
    temperature=0,
)
print("[+]  LLM Loaded")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

def read_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

try:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        pdf_text = read_from_pdf(uploaded_file) # Read the text from PDF file
        
        text = text_splitter.split_text(pdf_text) # Split the text into chunks by using text_splitter
        document_search = FAISS.from_texts(text, embeddings) # Convert each text chunks into vectors by using the embeddings and create a searchable index of text vectors.

        
        chain = load_qa_chain(llm, chain_type="stuff") # Create a question answering chain along with LLM and type stuff.

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("how can i help you"):
            # Display user message in chat message container

            st.chat_message("user").markdown(prompt) 

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            docs = document_search.similarity_search(prompt) # It takes prompt(which is user query) and converts it into vector using the same embedding model, It then searches FAISS vector store for text chunks similar to it.
            response = chain.run(input_documents=docs, question=prompt) # It takes the QA chain to process the input documents extracted from FAISS and the question to generate an answer.

            # Display assistant response in chat message container

            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            


except Exception as ex:
    st.exception(str(ex))
    print(str(ex))
