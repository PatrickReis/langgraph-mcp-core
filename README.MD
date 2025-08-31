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

**Ollama:**
- `OLLAMA_BASE_URL`: URL do servidor Ollama (padrão: http://localhost:11434)
- `OLLAMA_MODEL`: Modelo LLM principal (padrão: llama3:latest)
- `OLLAMA_EMBEDDINGS_MODEL`: Modelo para embeddings (padrão: nomic-embed-text)

**ChromaDB:**
- `CHROMA_PERSIST_DIRECTORY`: Diretório de persistência (padrão: ./chroma_db)

**Busca Vetorial:**
- `VECTOR_SEARCH_K_RESULTS`: Número de resultados da busca (padrão: 3)

**MCP Server:**
- `MCP_SERVER_NAME`: Nome do servidor MCP (padrão: LangGraphToolsMCP)

### 3. Iniciar Ollama
```bash
ollama serve
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

## 📁 Estrutura do Projeto

- `main.py`: Agente principal com interface de linha de comando
- `tools.py`: Ferramentas de busca vetorial
- `mcp_server.py`: Servidor MCP
- `mcp_bridge.py`: Bridge entre LangChain e MCP
- `.env`: Configurações do ambiente (não commitado)
- `.env.example`: Exemplo de configurações