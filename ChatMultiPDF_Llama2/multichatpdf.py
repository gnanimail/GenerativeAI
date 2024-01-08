import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.ctransformers import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def show_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def create_conversational_chain(vector_store):
    load_dotenv()
    # Create llm
    my_llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                           model_type="llama",
                           streaming=True, 
                           callbacks=[StreamingStdOutCallbackHandler()],
                           config={'max_new_tokens': 500, 'temperature': 0.01})
    
    # create memory for storing chat history
    my_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # create convesation retrieval chain 
    chain = ConversationalRetrievalChain.from_llm(llm=my_llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=my_memory)
    return chain


def chatmultipdf():
    
    # initialize session state 
    initialize_session_state()
    st.title("Virtual Assistant to chat Multiple PDFs")
    
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        # show chat history
        show_chat_history(chain)


if __name__ == "__main__":
    chatmultipdf()

