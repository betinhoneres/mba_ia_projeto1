# Projeto de Busca Semântica (RAG) Multi-LLM

Este projeto implementa uma ferramenta de Busca Semântica (RAG) capaz de processar documentos PDF e realizar consultas em linguagem natural via terminal (CLI). O sistema garante que as respostas sejam estritamente baseadas no conteúdo fornecido, utilizando o ecossistema LangChain, armazenamento vetorial no PostgreSQL (pgVector) e modelos da OpenAI e do Google Gemini.

## Pré-requisitos

* **Python:** Versão 3.10 ou superior (com ambiente virtual `venv`).
* **Docker e Docker Compose:** Para rodar o banco de dados PostgreSQL com a extensão pgVector.

### OBSERVAÇÕES IMPORTANTES
Foi criado dois arquivos para ingestão dos dados. Um para o gemini e outro para a openai. 
Para rodar os comandos seguintes (chat e search) deve ser adicionado o argumento para qual dos modelos está sendo solicitado.

Foi necessário alterar o modelo do google gemini para **gemini-embedding-001** porque o modelo **models/embedding-001** foi descontinuado e não está acessivel.

---

## Configuração do Ambiente (Fase 01)

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
#O modelo models/embedding-001 foi descontinuado e não está acessivel
GEMINI_CHAT_MODEL=gemini-2.5-flash-lite
```

### 2. Inicialização do Banco de Dados
Antes de rodar qualquer script, suba o container do banco de dados na porta 5432:  
```
docker-compose up -d
```
### 3. Como Executar o Projeto (Passo a Passo)
O arquivo document.pdf deve estar na raiz do projeto antes de iniciar o pipeline.  
#### Passo 1: Ingestão e Vetorização (Fase 02)   
Este processo lê o PDF, divide em fragmentos de 1000 caracteres e salva os vetores no banco de dados.  
Para processar via OpenAI: 
```
python src/ingest_openai.py
```
Para processar via Gemini: 
```
python src/ingest_gemini.py
```
#### Passo 2: Listagem de Fragmentos (Metadata Logging)   
Para garantir a transparência do pipeline de ingestão, você pode consultar as coleções e retornar os IDs e os primeiros 100 caracteres de cada chunk.  
```
python src/list_chunks.py --provider openai
```
```
python src/list_chunks.py --provider gemini
```
#### Passo 3: Teste do Core de Busca (Fase 03)   
Retorna os fragmentos de texto relevantes para a sua pergunta, direto do banco de dados.  
```
python src/search.py --provider openai --query "Sua pergunta aqui"
```
```
python src/search.py --provider gemini --query "Sua pergunta aqui"
```
#### Passo 4: Loop de Chat Interativo (Fase 04)   
Abre a interface CLI funcional de chat interativo. O sistema usará o contexto recuperado para formular respostas baseadas no PDF.  
```
python src/chat.py --provider openai
```
```
python src/chat.py --provider gemini
```
#### Restrição Crítica 
Se a informação não estiver no PDF, a resposta deve ser obrigatoriamente: 
**"Não tenho informações necessárias para responder sua pergunta."**. 
É proibido o uso de conhecimento externo ou opiniões.