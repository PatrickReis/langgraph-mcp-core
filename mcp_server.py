import os
from dotenv import load_dotenv
from logger.logger import get_logger
# Carregar variáveis de ambiente
load_dotenv()

# Obter loggers especializados
mcp_logger = get_logger("mcp")
app_logger = get_logger("app")

mcp_logger.info("🔧 Carregando servidor MCP...")

# Tentar usar MCP oficial primeiro (para mcp dev), senão FastMCP (para fastmcp run)
try:
    from mcp.server.fastmcp.server import FastMCP as OfficialFastMCP
    mcp = OfficialFastMCP(os.getenv("MCP_SERVER_NAME", "LangGraphToolsMCP"))
    using_official = True
    mcp_logger.info("Usando MCP oficial")
except ImportError:
    from fastmcp import FastMCP
    mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "LangGraphToolsMCP"))
    using_official = False
    mcp_logger.error("Usando FastMCP")


# Carregar tools do LangChain
mcp_logger.info("Carregando tools...")
try:
    from tools.tools import tools
    mcp_logger.info(f"{len(tools)} tools carregadas:")
    for tool in tools:
        mcp_logger.info(f"  - {tool.name}")
except Exception as e:
    mcp_logger.error(f"Erro ao carregar tools: {e}")
    tools = []

# Carregar e executar transform
mcp_logger.info("🌉 Executando transform...")
try:
    from transform.lang_mcp_transform import register_langchain_tools_as_mcp, register_langgraph_agent_as_mcp
    
    # Registrar tools individuais
    register_langchain_tools_as_mcp(mcp, tools)
    
    # Registrar agente completo do main.py
    mcp_logger.info("🤖 Registrando agente LangGraph completo...")
    from graphs.graph import create_agent_graph
    from providers.llm_providers import get_llm
    
    # Criar função wrapper para o agente
    def create_agent():
        return create_agent_graph(get_llm())
    
    register_langgraph_agent_as_mcp(
        mcp, 
        create_agent,
        agent_name="langgraph_orchestrator",
        description="Orquestrador LangGraph completo que decide automaticamente quando usar ferramentas específicas baseado na pergunta do usuário"
    )
    
    mcp_logger.info("transform concluído com sucesso")
except Exception as e:
    mcp_logger.error(f"Erro no transform: {e}")

# Registrar MCP Resources
mcp_logger.info("📚 Registrando MCP Resources...")
try:
    @mcp.resource("config://agent")
    async def get_agent_config():
        """Configuração do agente LangGraph"""
        from providers.llm_providers import get_provider_info
        provider_info = get_provider_info()
        
        return {
            "provider": provider_info.get("provider", "unknown"),
            "model": provider_info.get("model", "unknown"),
            "temperature": provider_info.get("temperature", 0.7),
            "max_tokens": provider_info.get("max_tokens", 2000),
            "tools_available": [tool.name for tool in tools],
            "capabilities": [
                "knowledge_base_search",
                "weather_queries", 
                "direct_chat",
                "tool_orchestration"
            ]
        }
    
    @mcp.resource("knowledge://base")  
    async def get_knowledge_base_info():
        """Informações sobre a base de conhecimento"""
        try:
            from tools.tools import tools
            search_tool = next((t for t in tools if t.name == "search_knowledge_base"), None)
            if search_tool:
                return {
                    "type": "ChromaDB Vector Store",
                    "description": "Base vetorial com documentos sobre programação, IA, LangGraph",
                    "keywords": [
                        "python", "programação", "langgraph", "chromadb", "ollama",
                        "machine learning", "ia", "inteligência artificial", "rag",
                        "deep learning", "nlp"
                    ],
                    "status": "available"
                }
            else:
                return {"status": "not_available", "reason": "search_knowledge_base tool not found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.resource("status://system")
    async def get_system_status():
        """Status geral do sistema"""
        from providers.llm_providers import get_provider_info
        return {
            "mcp_server": "running",
            "langgraph_agent": "available", 
            "tools_count": len(tools),
            "knowledge_base": "available",
            "llm_provider": get_provider_info().get("provider", "unknown")
        }
    
    mcp_logger.info("Resources registrados com sucesso")
except Exception as e:
    mcp_logger.error(f"Erro ao registrar resources: {e}")

# Registrar MCP Prompts
mcp_logger.info("💭 Registrando MCP Prompts...")
try:
    @mcp.prompt("agent-query")
    async def agent_query_prompt(topic: str = "", style: str = "conversational"):
        """Template para consultas ao agente LangGraph"""
        
        styles = {
            "conversational": "Responda de forma conversacional e amigável",
            "technical": "Forneça uma resposta técnica e detalhada", 
            "concise": "Seja conciso e direto ao ponto",
            "educational": "Explique como se estivesse ensinando"
        }
        
        system_msg = styles.get(style, styles["conversational"])
        user_content = f"Sobre {topic}: " if topic else "Como posso ajudar você hoje?"
        
        return f"""Instruções do sistema: {system_msg}. Use ferramentas quando necessário e sempre responda em português brasileiro.

Usuário: {user_content}"""
    
    @mcp.prompt("knowledge-search")
    async def knowledge_search_prompt(query: str):
        """Template para busca na base de conhecimento"""
        return f"""Instruções do sistema: Você deve buscar informações na base de conhecimento e fornecer uma resposta completa baseada no que encontrar. Sempre cite as fontes quando possível.

Usuário: Busque na base de conhecimento informações sobre: {query}"""
    
    @mcp.prompt("weather-query") 
    async def weather_query_prompt(location: str):
        """Template para consultas meteorológicas"""
        return f"""Instruções do sistema: Use a ferramenta de clima para obter informações meteorológicas atuais e forneça uma resposta clara sobre as condições.

Usuário: Qual é o clima atual em {location}?"""
    
    @mcp.prompt("tool-orchestration")
    async def tool_orchestration_prompt(task: str):
        """Template para tarefas que podem precisar de múltiplas ferramentas"""
        return f"""Instruções do sistema: Analise a tarefa e determine quais ferramentas são necessárias. Use múltiplas ferramentas se necessário para fornecer uma resposta completa.

Usuário: Tarefa: {task}"""
    
    mcp_logger.info("Prompts registrados com sucesso")
except Exception as e:
    mcp_logger.error(f"Erro ao registrar prompts: {e}")
    
# Iniciar servidor MCP
if __name__ == "__main__":
    mcp_logger.info("🚀 Iniciando servidor MCP...")
    try:
        mcp.run()
    except Exception as e:
        mcp_logger.error(f"❌ Erro ao iniciar servidor: {e}")