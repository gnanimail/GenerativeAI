import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.ctransformers import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
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
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
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



def vitualassistant_chain(knowledge):
    # create llm to generate response
    my_llm = CTransformers(model="model/mistral-7b-instruct-v0.1.Q5_0.gguf",
                           model_type="mistral",
                           config={"max_new_tokens":128,
                                   "temperature":0.2},
                           verbose=True
                           )
    
    # create memory to hold the chat history 
    my_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # create convrational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(llm=my_llm, chain_type='stuff',
                                                 retriever=knowledge.as_retriever(search_kwargs={"k": 2}),
                                                 memory=my_memory)
    return chain




def chatpdf():

    # initialize session state to fetch the old chat history
    initialize_session_state()
    st.title("Virtual Assistant to chat Multiple PDF")

    # Initialize Streamlit
    st.sidebar.title("Document Uploading")
    uplod_documents = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uplod_documents:
        #create content array
        content = []
        #look into all uploaded documents
        for document in uplod_documents:
            #get the filename and the file type
            document_extension = os.path.splitext(document.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(document.read())
                temp_file_path = temp_file.name

            loader = None
            if document_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                content.extend(loader.load())
                os.remove(temp_file_path)

        content_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        content_chunks = content_splitter.split_documents(content)

        # Create embeddings
        create_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                   model_kwargs={'device': 'cpu'})

        # Create vector store
        knowledge = FAISS.from_documents(content_chunks, embedding=create_embeddings)

        # Create the chain object
        chain = vitualassistant_chain(knowledge)

        # show the chat history
        show_chat_history(chain)



if __name__=="__main__":
    chatpdf()