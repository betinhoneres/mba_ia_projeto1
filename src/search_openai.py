import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Conexão ao PostgreSQL (mesmo padrão dos scripts de ingestão)
connection = (
    f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

collection_name = "pdf_openai"

def run_search_openai(query: str, k: int = 10):
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    vectorstore = PGVector(
        connection=connection,
        collection_name=collection_name,
        embedding=embeddings,
        use_jsonb=True,
    )

    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

if __name__ == "__main__":
    pergunta = input("Digite sua pergunta: ")
    resultados = run_search_openai(pergunta)

    print("\n=== RESULTADOS OPENAI ===")
    for doc, score in resultados:
        print(f"\nScore: {score}")
        print(doc.page_content)
