import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate

load_dotenv()

def run_chat(provider):
    conn_str = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    
    # 1. Configuração específica do Provedor [cite: 60-62]
    if provider == "openai":
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
        llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"), temperature=0.0)
        collection = "pdf_openai"
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL"))
        llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_CHAT_MODEL"), temperature=0.0)
        collection = "pdf_gemini"

    # 2. Configuração do Banco de Dados [cite: 39]
    vectorstore = PGVector(
        collection_name=collection,
        connection=conn_str,
        embeddings=embeddings,
    )
    
    # 3. Prompt Template Rígido [cite: 64-69]
    template = """
    CONTEXTO: {contexto}
    
    REGRAS:
    - Responda somente com base no CONTEXTO.
    - Se não houver informação, responda: "Não tenho informações necessárias para responder sua pergunta."
    - Proibido conhecimento externo ou opiniões.
    
    PERGUNTA: {pergunta}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    print(f"\n=== Chat Interativo Iniciado ({provider.upper()}) ===")
    print("Digite 'sair' a qualquer momento para encerrar o programa.\n")

    # 4. Loop de Chat Interativo (Fase 04) 
    while True:
        # Aguarda a entrada do usuário
        query = input(f"[{provider.upper()}] Sua pergunta: ")
        
        # Condição de saída
        if query.strip().lower() in ['sair', 'exit', 'quit']:
            print("Encerrando o chat. Até logo!")
            break
            
        # Ignora envios vazios
        if not query.strip():
            continue

        # Recuperação (k=10 conforme PRD) [cite: 11, 57]
        docs = vectorstore.similarity_search(query, k=10)
        context = "\n---\n".join([d.page_content for d in docs])
        
        # Geração da resposta
        response = chain.invoke({"contexto": context, "pergunta": query})
        
        print(f"\n[Resposta]:\n{response.content}\n")
        print("-" * 50) # Linha de separação visual

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Removemos o --query, mantendo apenas o provedor
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    args = parser.parse_args()
    
    run_chat(args.provider)