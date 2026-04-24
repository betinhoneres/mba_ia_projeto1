import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Carrega as variáveis do seu ficheiro .env
load_dotenv()

# Configuração da ligação baseada no seu Docker/PRP
connection = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
collection_name = "pdf_gemini"

def run_ingestion_gemini():
    print("Iniciando a ingestão com Gemini...")
    
    # 1. Carregamento do PDF (Especificação 3.2) 
    loader = PyPDFLoader("document.pdf")
    docs = loader.load()
    
    # 2. Fragmentação/Chunking (1000 caracteres, 150 overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(chunks)} fragmentos.")
    
    # 3. Configuração dos Embeddings do Google 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 4. Persistência no pgVector 
    try:
        PGVector.from_documents(
            embedding=embeddings,
            documents=chunks,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        print(f"Sucesso: Dados salvos na coleção '{collection_name}' no PostgreSQL.")
    except Exception as e:
        # Mitigação de risco: Lógica de erro para conectividade 
        print(f"Erro ao conectar ou salvar no banco: {e}")

if __name__ == "__main__":
    run_ingestion_gemini()