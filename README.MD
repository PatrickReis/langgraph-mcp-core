# LangGraph Agent com MCP Bridge

Este projeto implementa um agente inteligente usando LangGraph com acesso a uma base de conhecimento vetorial e integração MCP (Model Context Protocol).

## 🚀 Configuração

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
Copie o arquivo `.env.example` para `.env` e ajuste as configurações:

```bash
cp .env.example .env
```

#### Configurações disponíveis:

**Provedor Principal:**
- `MAIN_PROVIDER`: Provedor LLM principal ('ollama', 'openai', 'gemini') (padrão: ollama)

**Ollama:**
- `OLLAMA_BASE_URL`: URL do servidor Ollama (padrão: http://localhost:11434)
- `OLLAMA_MODEL`: Modelo LLM principal (padrão: llama3:latest)
- `OLLAMA_EMBEDDINGS_MODEL`: Modelo para embeddings (padrão: nomic-embed-text)

**OpenAI:**
- `OPENAI_API_KEY`: Sua chave de API da OpenAI
- `OPENAI_MODEL`: Modelo OpenAI (padrão: gpt-3.5-turbo)

**Google Gemini:**
- `GEMINI_API_KEY`: Sua chave de API do Google AI Studio
- `GEMINI_MODEL`: Modelo Gemini (padrão: gemini-1.5-flash)

**ChromaDB:**
- `CHROMA_PERSIST_DIRECTORY`: Diretório de persistência (padrão: ./chroma_db)

**Busca Vetorial:**
- `VECTOR_SEARCH_K_RESULTS`: Número de resultados da busca (padrão: 3)

**MCP Server:**
- `MCP_SERVER_NAME`: Nome do servidor MCP (padrão: LangGraphToolsMCP)

### 3. Configurar Provedor LLM

**Para Ollama (padrão):**
```bash
# O arquivo .env já está configurado para Ollama
ollama serve
```

**Para OpenAI:**
```bash
# Editar .env
MAIN_PROVIDER=openai
OPENAI_API_KEY=sua_chave_aqui

# Instalar dependência
pip install langchain-openai
```

**Para Google Gemini:**
```bash
# Editar .env
MAIN_PROVIDER=gemini
GEMINI_API_KEY=sua_chave_aqui

# Instalar dependência
pip install langchain-google-genai
```

### 4. Testar Provedores
```bash
python test_providers.py
```

## 🎯 Uso

### Executar o agente principal
```bash
python main.py
```

### Executar servidor MCP
```bash
fastmcp run mcp_server.py
```

### Executar servidor MCP com HTTP
```bash
fastmcp run mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8088

```

## 🔧 Funcionalidades

- **Agente Inteligente**: Processa perguntas usando LangGraph
- **Base de Conhecimento**: Busca vetorial com ChromaDB
- **Integração MCP**: Bridge para ferramentas LangChain
- **Configuração Flexível**: Todas as configurações via variáveis de ambiente
- **Múltiplos Provedores**: Suporte para Ollama, OpenAI e Google Gemini
- **Arquitetura Modular**: Sistema componentizado para fácil extensão

## 📁 Estrutura do Projeto

- `main.py`: Agente principal com interface de linha de comando
- `tools.py`: Ferramentas de busca vetorial
- `mcp_server.py`: Servidor MCP
- `mcp_bridge.py`: Bridge entre LangChain e MCP
- `llm_providers.py`: Sistema modular de provedores LLM
- `.env`: Configurações do ambiente (não commitado)
- `.env.example`: Exemplo de configurações