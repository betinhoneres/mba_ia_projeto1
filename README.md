# Projeto de Busca Semântica (RAG) Multi-LLM

Este projeto implementa uma ferramenta de Busca Semântica (RAG) capaz de processar documentos PDF e realizar consultas em linguagem natural via terminal (CLI). O sistema garante que as respostas sejam estritamente baseadas no conteúdo fornecido, utilizando o ecossistema LangChain, armazenamento vetorial no PostgreSQL (pgVector) e modelos da OpenAI e do Google Gemini.

## 🛠️ Pré-requisitos

* **Python:** Versão 3.10 ou superior (com ambiente virtual `venv`).
* **Docker e Docker Compose:** Para rodar o banco de dados PostgreSQL com a extensão pgVector.

---

## ⚙️ Configuração do Ambiente (Fase 01)

### 1. Variáveis de Ambiente (`.env`)
O projeto exige a configuração de credenciais e parâmetros de modelos para funcionar. Crie um arquivo chamado `.env` na raiz do projeto com a seguinte estrutura:

```env
# --- CONFIGURAÇÕES DO BANCO DE DADOS (DOCKER) ---
DB_USER=adminpg
DB_PASS=Postgres123
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=dbvector

# --- CHAVES DAS APIS (Obrigatório) ---
OPENAI_API_KEY=sua_chave_da_openai_aqui
GOOGLE_API_KEY=sua_chave_do_google_aqui

# --- CONFIGURAÇÕES DE MODELOS ---
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5-nano
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-2.5-flash-lite

# --- OBSERVAÇÕES IMPORTANTES ---
Foi necessário alterar o modelo do google gemini para gemini-embedding-001 porque o modelo models/embedding-001 foi descontinuado e não está acessivel.