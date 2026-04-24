import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

# 1. Carrega as variáveis do ficheiro .env localizado na raiz
load_dotenv()

# Configuração da ligação ao PostgreSQL (Docker)
# As variáveis são lidas do seu .env conforme a Especificação Técnica 3.1
connection = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# Nome da coleção específica para isolar os dados da OpenAI 
collection_name = "pdf_openai"

def run_ingestion_openai():
    print("Iniciando a ingestão com OpenAI...")
    
    # 2. Carregamento do PDF (Especificação Técnica 3.2)
    pdf_path = "document.pdf"
    if not os.path.exists(pdf_path):
        print(f"Erro: O ficheiro {pdf_path} não foi encontrado na raiz do projeto.")
        return

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 3. Fragmentação (Chunking) conforme o seu PRP
    # chunk_size: 1000, chunk_overlap: 150
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(chunks)} fragmentos.")
    
    # 4. Configuração dos Embeddings (Especificação Técnica 3.2)
    # Modelo: text-embedding-3-small
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    
    # 5. Persistência no pgVector (Fase 02 do Cronograma)
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
        # Tratamento de erro de conectividade com o Docker
        print(f"Erro ao conectar ou guardar no banco de dados: {e}")

if __name__ == "__main__":
    run_ingestion_openai()