import os
import argparse
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import create_engine, text

load_dotenv()

def list_chunks(provider):
    conn_str = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    collection = f"pdf_{provider}"
    
    print(f"--- Listando fragmentos da coleção: {collection} ---")
    
    engine = create_engine(conn_str)
    # Consulta direta para buscar metadados e conteúdo
    query = text("""
        SELECT c.uuid, c.document, c.cmetadata 
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = :name
        LIMIT 5
    """)
    
    # Nota: Usamos SQL direto para simplificar a visualização dos primeiros 100 caracteres
    with engine.connect() as conn:
        # Busca simplificada via LangChain para validar os objetos
        embeddings = (OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")) if provider == "openai" 
                      else GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")))
        
        vectorstore = PGVector(
            collection_name=collection,
            connection=conn_str,
            embeddings=embeddings,
        )
        
        # Simulando a listagem via recuperação
        all_docs = vectorstore.similarity_search("", k=100)
        for doc in all_docs:
            content = doc.page_content.replace('\n', ' ')[:100]
            idx = doc.metadata.get('start_index', 'N/A')
            print(f"[ID: {idx}] Conteúdo: {content}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    args = parser.parse_args()
    list_chunks(args.provider)