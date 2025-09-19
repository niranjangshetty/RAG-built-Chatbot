import os
import google.generativeai as genai
from langchain.vectorstores import FAISS # This will be the vector database
from langchain_community.embeddings import HuggingFaceEmbeddings # To perform word embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # To split the text into chunks
from pypdf import PdfReader
import faiss
import streamlit as st

from pdfextractor import text_extractor_pdf

# Create the main page
st.title(":blue[AI-Powered Chatbot using RAG]")
tips = '''Follow the steps to use this application:
* Upload your PDF document in the sidebar.
* Write your query and start chatting with the bot.'''
st.subheader(tips)

# Load PDF in sidebar
st.sidebar.title(":orange[UPLOAD YOUR DOCUMENT HERE(PDF only)]")
file_uploaded = st.sidebar.file_uploader("Upload your file")
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)

    # Step-1 : Configure the Models

    # Configure LLM

    key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Configure Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    # Step-2 : Chunking (Create Chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    chunks = splitter.split_text(file_text)

    # Step-3 : Create FAISS Vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # Step-4 : Configure retreiver
    retriever = vector_store.as_retriever(search_kwargs={"k":5})

    # Lets create a function that will take the user query and return the response
    def generate_response(query):
        # Step-6 : Retrieval (R)
        retrieved_docs = retriever.get_relevant_documents(query= query)
        context = ' '.join([doc.page_content for doc in retrieved_docs])

        # Step-7 : Augmentation (A)
        prompt = f'''You are a helpful assistant using RAG.
        Context: {context}
        User Query: {query}
        Please provide a detailed response based on the context and the user query.
        '''
        # Step-8 : Generation (G)
        content = llm_model.generate_content(prompt).text
        return content

    # Step-5 : Creating a Chatbot to start the Conversation
    # Initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Display History
    for chat in st.session_state['history']:
        if chat['role'] == 'user':
            st.write(f":orange[User:] {chat['text']}")
        else:
            st.write(f":green[Chatbot:] {chat['text']}")

    # Input from the user (using Streamlit form)
    with st.form('chat_form', clear_on_submit=True):
        user_input = st.text_input("Enter your query here: ")
        submit_button = st.form_submit_button("Send")

    # start the conversation and append the output and query to the history
    if user_input and submit_button:

        st.session_state['history'].append({'role':'user', 'text':user_input})

        model_output = generate_response(user_input)
        st.session_state['history'].append({'role':'Chatbot', 'text':model_output})

        st.rerun()