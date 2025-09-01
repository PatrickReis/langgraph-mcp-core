"""
Agent core - Clean Architecture Implementation
Professional main entry point using Clean Architecture patterns.
"""

from core.entities.agent import AgentConfig, ProviderType
from core.use_cases.agent_orchestration import AgentOrchestrationUseCase
from adapters.llm.factory import LLMProviderFactory
from infrastructure.config.settings import settings
from shared.utils.logger import get_logger

# Get specialized loggers
app_logger = get_logger("app")
agent_logger = get_logger("agent")


def create_agent_orchestrator() -> AgentOrchestrationUseCase:
    """
    Create agent orchestrator using Clean Architecture.
    
    Returns:
        Configured AgentOrchestrationUseCase
    """
    app_logger.info("Criando orquestrador de agente com Clean Architecture")
    
    # Criar provedor LLM
    try:
        provider_type = ProviderType(settings.llm.main_provider)
        llm_provider = LLMProviderFactory.create_provider(provider_type)
        app_logger.success("Provedor LLM criado", provider=settings.llm.main_provider)
    except Exception as e:
        app_logger.error(f"Falha ao criar provedor LLM: {e}")
        raise
    
    # Criar configuração do agente
    config = AgentConfig(
        name="Agent core Agent",
        provider=provider_type,
        model=getattr(settings.llm, f"{settings.llm.main_provider}_model"),
        temperature=0.7
    )
    
    # Criar embeddings e armazenamento vetorial
    from adapters.llm.providers import create_embeddings_provider
    from adapters.storage.vector_store_adapter import ChromaVectorStoreAdapter
    from adapters.tools.langchain_tool_repository import LangChainToolRepository
    
    try:
        # Inicializar embeddings
        embeddings = create_embeddings_provider(provider_type)
        app_logger.success("Provedor de embeddings criado")
        
        # Inicializar armazenamento vetorial
        storage_adapter = ChromaVectorStoreAdapter(embeddings)
        vector_store = storage_adapter.get_vector_store()
        app_logger.success("Armazenamento vetorial inicializado")
        
        # Inicializar repositório de ferramentas com armazenamento vetorial
        tool_repository = LangChainToolRepository(vector_store)
        app_logger.success("Repositório de ferramentas inicializado")
        
    except Exception as e:
        app_logger.error(f"Falha ao inicializar armazenamento/ferramentas: {e}")
        raise
    
    # Criar orquestrador
    orchestrator = AgentOrchestrationUseCase(
        llm_provider=llm_provider,
        tool_repository=tool_repository,
        config=config
    )
    
    agent_logger.success("Orquestrador de agente criado com sucesso")
    return orchestrator


def interactive_mode():
    """Run in interactive mode."""
    app_logger.info("Iniciando Agent core em modo interativo")
    
    try:
        orchestrator = create_agent_orchestrator()
        
        print(f"\n🤖 {settings.app_name} v{settings.version}")
        print("Digite suas perguntas (ou 'sair' para encerrar):\n")
        
        while True:
            try:
                user_input = input("👤 Você: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'sair', 'parar']:
                    app_logger.info("Sessão interativa encerrada pelo usuário")
                    print("👋 Até logo!")
                    break
                
                if not user_input:
                    continue
                
                # Executar consulta usando Clean Architecture
                execution = orchestrator.execute_query(user_input)
                
                if execution.success:
                    print(f"🤖 Agente: {execution.response}")
                    
                    if execution.tools_used:
                        agent_logger.info(f"Ferramentas utilizadas: {', '.join(execution.tools_used)}")
                else:
                    print(f"❌ Erro: {execution.error_message}")
                    
                print()  # Linha vazia para legibilidade
                
            except KeyboardInterrupt:
                app_logger.info("Sessão interativa interrompida")
                print("\n👋 Até logo!")
                break
            except Exception as e:
                app_logger.error(f"Erro durante execução do agente: {str(e)}")
                print(f"❌ Erro: {str(e)}\n")
                
    except Exception as e:
        app_logger.error(f"Falha ao iniciar modo interativo: {e}")
        print(f"❌ Erro de Inicialização: {e}")


if __name__ == "__main__":
    interactive_mode()