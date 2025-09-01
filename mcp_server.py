"""
MCP Server - Clean Architecture Implementation
Professional MCP server using Clean Architecture patterns.
"""

from fastmcp import FastMCP
from core.entities.agent import AgentConfig, ProviderType
from core.use_cases.agent_orchestration import AgentOrchestrationUseCase
from adapters.llm.factory import LLMProviderFactory
from adapters.llm.providers import create_embeddings_provider
from adapters.storage.vector_store_adapter import ChromaVectorStoreAdapter
from adapters.tools.langchain_tool_repository import LangChainToolRepository
from adapters.mcp.mcp_server_adapter import MCPServerAdapter
from infrastructure.config.settings import settings
from shared.utils.logger import get_logger

# Obter loggers especializados
mcp_logger = get_logger("mcp")
app_logger = get_logger("app")

# Tentar usar MCP oficial primeiro (para mcp dev), senão FastMCP (para fastmcp run)
try:
    from mcp.server.fastmcp.server import FastMCP as OfficialFastMCP
    mcp = OfficialFastMCP(settings.mcp.server_name)
    using_official = True
    mcp_logger.success("Using official MCP server")
except ImportError:
    from fastmcp import FastMCP
    mcp = FastMCP(settings.mcp.server_name)
    using_official = False
    mcp_logger.success("Using FastMCP server")


def create_agent_orchestrator() -> AgentOrchestrationUseCase:
    """Criar orquestrador de agente"""
    mcp_logger.info("Criando orquestrador de agente ")
    
    # Criar provedor LLM
    try:
        provider_type = ProviderType(settings.llm.main_provider)
        llm_provider = LLMProviderFactory.create_provider(provider_type)
        mcp_logger.success("Provedor LLM criado", provider=settings.llm.main_provider)
    except Exception as e:
        mcp_logger.error(f"Falha ao criar provedor LLM: {e}")
        raise
    
    # Criar configuração do agente
    config = AgentConfig(
        name="Agent core MCP Agent",
        provider=provider_type,
        model=getattr(settings.llm, f"{settings.llm.main_provider}_model"),
        temperature=0.7
    )
    
    # Criar embeddings e vector store
    try:
        # Inicializar embeddings
        embeddings = create_embeddings_provider(provider_type)
        mcp_logger.success("Provedor de embeddings criado")
        
        # Inicializar vector store
        vector_store = ChromaVectorStoreAdapter(embeddings=embeddings)
        mcp_logger.success("Vector store configurado")
    except Exception as e:
        mcp_logger.error(f"Falha ao configurar embeddings/vector store: {e}")
        raise
    
    # Criar repositório de ferramentas
    tool_repository = LangChainToolRepository()
    mcp_logger.success("Repositório de ferramentas criado")
    
    # Criar orquestrador
    orchestrator = AgentOrchestrationUseCase(
        llm_provider=llm_provider,
        tool_repository=tool_repository,
        config=config
    )
    
    mcp_logger.success("Orquestrador de agente criado com sucesso")
    return orchestrator

# Configurar ferramentas MCP (necessário para mcp dev)
def setup_mcp_tools():
    """Configurar todas as ferramentas MCP"""
    try:
        mcp_logger.info("Configurando ferramentas MCP com Clean Architecture")
        
        # Criar orquestrador de agente
        orchestrator = create_agent_orchestrator()
        
        # Criar adaptador MCP
        mcp_adapter = MCPServerAdapter(mcp)
        
        # Registrar orquestrador como ferramenta MCP
        mcp_adapter.register_agent_as_tool(
            agent=orchestrator,
            tool_name="ai_accelerator",
            description="Agent core agent with knowledge base and tool access"
        )
        
        # Registrar ferramentas utilitárias adicionais
        @mcp.tool("Obter status do sistema e configuração")
        def get_system_status() -> str:
            """Obter status atual do sistema."""
            try:
                info = {
                    "app_name": settings.app_name,
                    "version": settings.version,
                    "environment": settings.environment,
                    "llm_provider": settings.llm.main_provider,
                    "llm_model": getattr(settings.llm, f"{settings.llm.main_provider}_model"),
                    "vector_store": settings.vector_store.collection_name
                }
                
                return f"Status do Sistema:\n" + "\n".join([f"- {k}: {v}" for k, v in info.items()])
                
            except Exception as e:
                mcp_logger.error(f"Erro ao obter status do sistema: {e}")
                return f"Erro: {str(e)}"
        
        @mcp.tool("Testar conectividade e configuração do agente")
        def test_agent_connection() -> str:
            """Testar conexões do agente e provedores."""
            try:
                # Testar provedor LLM
                provider_type = ProviderType(settings.llm.main_provider)
                llm_provider = LLMProviderFactory.create_provider(provider_type)
                
                if llm_provider.test_connection():
                    connection_status = "✅ Provedor LLM: Conectado"
                else:
                    connection_status = "❌ Provedor LLM: Falhou"
                
                # Testar funcionalidade básica
                test_response = llm_provider.generate_response("Olá, responda com 'Teste bem-sucedido'")
                
                return f"Resultados do Teste de Conexão:\n{connection_status}\nResposta do Teste: {test_response[:100]}"
                
            except Exception as e:
                mcp_logger.error(f"Teste de conexão falhou: {e}")
                return f"Teste de conexão falhou: {str(e)}"
        
        mcp_logger.success("Configuração das ferramentas MCP concluída com sucesso")
        return mcp_adapter
        
    except Exception as e:
        mcp_logger.error(f"Falha ao configurar ferramentas MCP: {e}")
        raise

# Configurar as ferramentas no nível global para mcp dev
setup_mcp_tools()






if __name__ == "__main__":
    try:
        mcp_logger.info("Iniciando Agent core MCP Server")
        mcp_logger.info(f"Configuração: {settings.llm.main_provider} na porta {settings.mcp.port}")
        
        # Configurar ferramentas
        mcp_adapter = setup_mcp_tools()
        
        # Iniciar servidor
        mcp_logger.info("MCP Server pronto - use 'mcp dev mcp_server_clean.py' para testar")
        mcp.run(transport=settings.mcp.transport)
        
    except Exception as e:
        mcp_logger.error(f"Falha ao iniciar MCP server: {e}")
        raise