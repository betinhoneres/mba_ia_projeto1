import os
import argparse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Carrega as variáveis do arquivo .env
load_dotenv()

def run_search(provider, query):
    # Configuração da conexão com o banco
    conn_str = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    
    # Seleção de provedor e embeddings
    if provider == "openai":
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
        collection = "pdf_openai"
    else:
        # Usa o modelo do Gemini atualizado no seu .env
        embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL"))
        collection = "pdf_gemini"

    # Conexão com o banco vetorial
    vectorstore = PGVector(
        collection_name=collection,
        connection=conn_str,
        embeddings=embeddings,
    )
    
    print(f"\n=== Iniciando Busca Semântica ({provider.upper()}) ===")
    print(f"Buscando por: '{query}'")
    print("=" * 50)
    
    # Recuperação: Método similarity_search_with_score com k=10 
    results = vectorstore.similarity_search_with_score(query, k=10)
    
    if not results:
        print("Nenhum fragmento encontrado.")
        return

    # Exibindo os fragmentos relevantes (Entregável da Fase 03) 
    for i, (doc, score) in enumerate(results, 1):
        # O score de distância varia dependendo da métrica (L2, Cosseno, etc.)
        # Menor distância normalmente indica maior similaridade semântica
        print(f"\n[{i}/10] Distância (Score): {score:.4f}")
        # Exibimos os primeiros 300 caracteres para não poluir muito a tela
        conteudo_limpo = doc.page_content.replace('\n', ' ')
        print(f"Conteúdo: {conteudo_limpo[:300]}...\n")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    # Adicionamos o argumento query para receber a pergunta via linha de comando
    parser.add_argument("--query", type=str, required=True, help="Texto para buscar")
    args = parser.parse_args()
    
    run_search(args.provider, args.query)