# 🤖 Agent core - Acelerador de Projetos de IA Generativa

**Template profissional para desenvolvimento de projetos de IA generativa usando Clean Architecture.**

Este projeto implementa um agente inteligente completo com arquitetura limpa, integração MCP (Model Context Protocol), base de conhecimento vetorial e sistema de ferramentas extensível.

## ✨ Características

- 🏗️ **Clean Architecture**: Estrutura profissional seguindo padrões da indústria
- 🔄 **Multi-Provider**: Suporte para Ollama, OpenAI e Google Gemini
- 🧠 **Base de Conhecimento**: Sistema vetorial com ChromaDB para RAG
- 🛠️ **Sistema de Ferramentas**: Integração com LangChain tools (web search, Wikipedia, calculadora)
- 📡 **MCP Integration**: Servidor MCP para integração com Claude Desktop
- 📝 **Logging Profissional**: Sistema de logs estruturado com métodos especializados para ferramentas e agentes
- ⚙️ **Configuração Centralizada**: Gerenciamento via variáveis de ambiente
- 🧪 **Testes Integrados**: Scripts de teste para todos os componentes

## 🏗️ Arquitetura Clean

```
📁 Agent core/
├── 🎯 core/                    # Camada de Negócio (Domain)
│   ├── entities/               # Entidades de domínio
│   │   ├── agent.py           # Modelos do agente
│   │   └── message.py         # Modelos de mensagem
│   ├── use_cases/             # Casos de uso (lógica de negócio)
│   │   └── agent_orchestration.py
│   └── interfaces/            # Contratos (abstrações)
│       ├── llm_provider.py
│       ├── tool_repository.py
│       ├── storage_repository.py
│       └── agent_interface.py
├── 🔌 adapters/               # Camada de Adaptadores
│   ├── llm/                   # Provedores LLM
│   │   ├── providers.py       # Implementações (Ollama, OpenAI, Gemini)
│   │   └── factory.py         # Factory pattern
│   ├── tools/                 # Ferramentas
│   │   └── langchain_tool_repository.py
│   ├── storage/               # Armazenamento
│   │   └── vector_store_adapter.py
│   └── mcp/                   # MCP Integration
│       └── mcp_server_adapter.py
├── 🏢 infrastructure/         # Camada de Infraestrutura
│   └── config/
│       └── settings.py        # Configurações centralizadas
├── 🔧 shared/                 # Utilitários compartilhados
│   └── utils/
│       └── logger.py          # Sistema de logging profissional com métodos especializados
├── 🚀 Entry Points           # Pontos de entrada
│   ├── main_clean.py         # CLI com Clean Architecture
│   ├── mcp_server_clean.py   # MCP Server com Clean Architecture
│   ├── main.py               # CLI legado (compatibilidade)
│   └── mcp_server.py         # MCP Server legado
└── 📊 logs/                  # Logs da aplicação
```

## 🚀 Configuração

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

### 3. Configurações Disponíveis

**🤖 Aplicação:**
- `APP_NAME`: Nome da aplicação (padrão: Agent core)
- `APP_VERSION`: Versão (padrão: 1.0.0)
- `ENVIRONMENT`: Ambiente (development/production)
- `DEBUG`: Modo debug (true/false)

**🧠 Provedor LLM:**
- `MAIN_PROVIDER`: Provedor principal ('ollama', 'openai', 'gemini')
- `OLLAMA_BASE_URL`: URL do Ollama (padrão: http://localhost:11434)
- `OLLAMA_MODEL`: Modelo Ollama (padrão: llama3:latest)
- `OLLAMA_EMBEDDINGS_MODEL`: Modelo embeddings (padrão: nomic-embed-text)
- `OPENAI_API_KEY`: Chave API OpenAI
- `OPENAI_MODEL`: Modelo OpenAI (padrão: gpt-3.5-turbo)
- `GEMINI_API_KEY`: Chave API Gemini
- `GEMINI_MODEL`: Modelo Gemini (padrão: gemini-1.5-flash)

**💾 Base de Conhecimento:**
- `CHROMA_PERSIST_DIRECTORY`: Diretório ChromaDB (padrão: ./data/vector_stores)
- `VECTOR_SEARCH_K_RESULTS`: Número de resultados (padrão: 3)
- `VECTOR_COLLECTION_NAME`: Nome da coleção (padrão: ai_accelerator_knowledge)

**📡 MCP Server:**
- `MCP_SERVER_NAME`: Nome do servidor (padrão: AgentCoreMCP)
- `MCP_HOST`: Host (padrão: 127.0.0.1)
- `MCP_PORT`: Porta (padrão: 8088)
- `MCP_TRANSPORT`: Transporte (padrão: sse)

**📝 Logging:**
- `LOG_LEVEL`: Nível de log (INFO, DEBUG, ERROR)
- `LOG_DIR`: Diretório dos logs (padrão: ./logs)
- `LOG_MAX_FILE_SIZE`: Tamanho máximo do arquivo (padrão: 10MB)
- `LOG_BACKUP_COUNT`: Número de backups (padrão: 5)

**Sistema de Logging Avançado:**
O logger inclui métodos especializados:
- `tool_execution()`: Para logs de execução de ferramentas
- `tool_success()` / `tool_error()`: Para resultados de ferramentas
- `knowledge_search()`: Para buscas na base de conhecimento
- `agent_decision()`: Para decisões do agente
- `success()` / `progress()`: Para indicadores visuais

## 🎯 Guia de Uso

### 🚀 Quick Start

**1. Executar CLI com Clean Architecture:**
```bash
python main_clean.py
```

**2. Executar MCP Server com Clean Architecture:**
```bash
mcp dev mcp_server_clean.py
```

### ⚙️ Configuração de Provedores

**Para Ollama (Recomendado para desenvolvimento):**
```bash
# 1. Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Iniciar servidor
ollama serve

# 3. Baixar modelos
ollama pull llama3:latest
ollama pull nomic-embed-text

# 4. Configurar .env
MAIN_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest
```

**Para OpenAI:**
```bash
# 1. Configurar .env
MAIN_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo

# 2. Instalar dependência
pip install langchain-openai
```

**Para Google Gemini:**
```bash
# 1. Configurar .env
MAIN_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash

# 2. Instalar dependência
pip install langchain-google-genai
```

### 🧪 Testes e Validação

**Testar configuração completa:**
```bash
# Testar provedores LLM
python -c "from infrastructure.config.settings import settings; print('✅ Configuração válida')"

# Testar CLI Clean Architecture
python main_clean.py
# Digite: "olá, como você está?"

# Testar MCP Server
mcp dev mcp_server_clean.py
```

**Usar MCP Inspector (Interface Web):**
```bash
# 1. Instalar MCP Inspector
npm install -g @mcp/inspector

# 2. Em terminal separado, executar MCP server
mcp dev mcp_server_clean.py

# 3. Em outro terminal, executar inspector
mcp-inspector

# 4. Abrir http://localhost:3000 no navegador
# 5. Conectar em: stdio ou sse://127.0.0.1:8088
```

### 🛠️ Ferramentas Disponíveis

O sistema inclui as seguintes ferramentas integradas:

- **🔍 search_knowledge_base**: Busca na base de conhecimento vetorial
- **🌐 web_search**: Busca na web usando DuckDuckGo
- **📚 wikipedia_search**: Busca na Wikipedia
- **🧮 calculator**: Calculadora matemática segura
- **⚙️ get_system_status**: Status do sistema e configurações
- **🔗 test_agent_connection**: Teste de conectividade dos provedores

### 📖 Exemplos de Uso

**Interação básica:**
```
👤 Você: Qual é a capital do Brasil?
🤖 Agent: A capital do Brasil é Brasília. [Ferramentas usadas: web_search]
```

**Busca com múltiplas ferramentas:**
```
👤 Você: Pesquise sobre machine learning e calcule 2+2
🤖 Agent: Machine learning é... [base de conhecimento] 
         Resultado do cálculo: 4 
         [Ferramentas usadas: search_knowledge_base, calculator]
```

## 🏗️ Arquitetura e Extensibilidade

### 📋 Princípios da Clean Architecture

1. **🎯 Independência de Frameworks**: Core não depende de bibliotecas externas
2. **🧪 Testabilidade**: Lógica de negócio isolada e testável
3. **🔄 Independência de UI**: Múltiplos pontos de entrada (CLI, MCP, API)
4. **💾 Independência de Banco**: Adapters abstraem persistência
5. **🌐 Independência Externa**: Provedores LLM intercambiáveis

### 🔧 Como Estender

**Adicionar novo provedor LLM:**
```python
# 1. Implementar interface em adapters/llm/providers.py
class NovoProvider(LLMProviderInterface):
    def generate_response(self, prompt: str) -> str:
        # Implementação específica
        pass

# 2. Adicionar ao factory em adapters/llm/factory.py
# 3. Adicionar enum em core/entities/agent.py
# 4. Configurar em infrastructure/config/settings.py
```

**Adicionar nova ferramenta:**
```python
# Em adapters/tools/langchain_tool_repository.py
class NovaFerramenta(BaseTool):
    name = "nova_ferramenta"
    description = "Descrição da ferramenta"
    
    def _run(self, query: str) -> str:
        # Implementação da ferramenta
        return "Resultado"
```

## 🔍 Monitoramento e Logs

O sistema possui um **sistema de logging profissional** com métodos especializados:

### 📝 Categorias de Logs
- **📱 app**: Logs gerais da aplicação
- **🤖 agent**: Logs do agente e orquestração
- **🛠️ tools**: Logs de execução de ferramentas (com `tool_execution`, `tool_success`, `tool_error`)
- **🧠 llm**: Logs dos provedores LLM
- **📡 mcp**: Logs do servidor MCP
- **💾 storage**: Logs da base de conhecimento

### 🎨 Métodos de Logging Especializados
```python
# Exemplos de uso do logger
logger.tool_execution("search_knowledge_base", query="python")
logger.tool_success("web_search", "Resultados encontrados")
logger.tool_error("calculator", "Expressão inválida")
logger.knowledge_search("machine learning")
logger.agent_decision("Escolhendo ferramenta web_search")
logger.success("Sistema inicializado com sucesso")
logger.progress("Processando consulta...")
```

### ⚙️ Configuração de Logs
```bash
# No .env
LOG_LEVEL=DEBUG  # Para desenvolvimento
LOG_LEVEL=INFO   # Para produção
```

### 📊 Formato dos Logs
- **Console**: `HH:MM:SS | LEVEL | component | message | context`
- **Arquivo**: `YYYY-MM-DD HH:MM:SS | LEVEL | component | file:line | message | context`

## 🚀 Deploy e Produção

### Docker (Recomendado)
```dockerfile
# Dockerfile exemplo
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main_clean.py"]
```

### Variáveis de Ambiente para Produção
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
LOG_ENABLE_CONSOLE=false
LOG_ENABLE_FILE=true
```

## 🔧 Resolução de Problemas

### ❌ ModuleNotFoundError: No module named 'logger_config'
**Solução**: Este erro foi corrigido! O projeto agora usa o sistema de logging unificado em `shared/utils/logger.py`. Se ainda ocorrer, verifique se todos os imports estão corretos:
```python
# ✅ Correto
from shared.utils.logger import get_logger
logger = get_logger("component")

# ❌ Incorreto (removido)
from logger_config import logger
```

### 🔍 'AIAcceleratorLogger' object has no attribute 'tool_error'
**Solução**: Todos os métodos necessários foram adicionados ao logger:
- `tool_error()`: Para erros de ferramentas
- `tool_success()`: Para sucessos de ferramentas
- Métodos especializados para diferentes contextos

### 🛠️ Desenvolvendo Novas Funcionalidades
Para garantir compatibilidade com o sistema de logging:
```python
# Importe o logger corretamente
from shared.utils.logger import get_logger
logger = get_logger("seu_componente")

# Use métodos apropriados
logger.tool_execution("nome_ferramenta", parametros={"key": "value"})
logger.tool_success("nome_ferramenta", "Operação realizada")
logger.error("Erro específico", contexto={"detalhes": "info"})
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

### 📋 Padrões de Código
- **Logger**: Sempre use `get_logger("componente")` do módulo `shared.utils.logger`
- **Arquitetura**: Siga os princípios da Clean Architecture
- **Testes**: Inclua testes para novas funcionalidades
- **Documentação**: Atualize o README quando necessário

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo `LICENSE` para detalhes.

---

**🎯 Agent core - Acelere seus projetos de IA generativa com arquitetura profissional!**