import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

connection = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
collection_name = "pdf_gemini"

def run_ingestion_gemini():
    print("Iniciando a ingestão com Gemini...")
    
    loader = PyPDFLoader("document.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(chunks)} fragmentos.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Processamento em lotes para evitar o erro 429 (Resource Exhausted)
    batch_size = 5 # Envia de 5 em 5 fragmentos
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            PGVector.from_documents(
                embedding=embeddings,
                documents=batch,
                collection_name=collection_name,
                connection=connection,
                use_jsonb=True,
            )
            print(f"Lote {i//batch_size + 1} salvo com sucesso...")
            time.sleep(2) # Pausa de 2 segundos entre lotes para respeitar a cota
        except Exception as e:
            print(f"Erro no lote {i//batch_size + 1}: {e}")
            break

    print(f"Ingestão finalizada na coleção '{collection_name}'.")

if __name__ == "__main__":
    run_ingestion_gemini()