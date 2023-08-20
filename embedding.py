from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DB_FAISS_PATH = 'vectorstore/db_faiss'

def generate_and_save_embeddings(data):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db




