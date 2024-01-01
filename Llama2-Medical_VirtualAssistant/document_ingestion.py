from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import faiss


DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore/db_faiss"

# create vector store database or knowledge base
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf",
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    
    medical_document = RecursiveCharacterTextSplitter(chunk_size=500,
                                                      chunk_overlap=50)
    
    #split the document
    text_chunks = medical_document.split_documents(documents)

    # create embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                          model_kwargs={"device":"cpu"})
    
    # store the embeddings into vector DB - FAISS
    knowledge_base = faiss.FAISS.from_documents(text_chunks, embeddings)
    knowledge_base.save_local(VECTORSTORE_PATH)


if __name__ == "__main__":
    create_vector_db()
