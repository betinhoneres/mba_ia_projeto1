import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

connection = (
    f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

collection_name = "pdf_gemini"

def run_search_gemini(query: str, k: int = 10):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
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
    resultados = run_search_gemini(pergunta)

    print("\n=== RESULTADOS GEMINI ===")
    for doc, score in resultados:
        print(f"\nScore: {score}")
        print(doc.page_content)
