import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import operator
from dotenv import load_dotenv
from providers.llm_providers import get_llm, get_embeddings, get_provider_info
from tools.tools import tools
from logger.logger import get_logger
from graphs.graph import create_agent_graph
# Carregar variáveis de ambiente
load_dotenv()

# Get specialized loggers
agent_logger = get_logger("agent")



# Configuração dos provedores LLM
try:
    llm = get_llm()
    embeddings = get_embeddings()
    provider_info = get_provider_info()
    agent_logger.success(f"Provedor LLM configurado: {provider_info['provider']}")
except Exception as e:
    agent_logger.error(f"Erro ao configurar provedor LLM: {e}")
    exit(1)



# Função principal para interagir com o agente
def run_agent():
    """Executa o agente em um loop interativo"""
    agent = create_agent_graph(llm)
    
    agent_logger.info("Agente LangGraph inicializado!")  
    agent_logger.info("Base vetorial ChromaDB carregada")
    agent_logger.info(f"Provedor LLM: {provider_info['provider']}")
    agent_logger.info(f"Modelo: {provider_info.get('model', 'N/A')}")
    agent_logger.info("\nDigite suas perguntas (ou 'quit' para sair):\n")
    
    while True:
        try:
            user_input = input("👤 Você: ").strip()
            
            if user_input.lower() in ['quit', 'sair', 'exit']:
                print("👋 Tchau!")
                break
            
            if not user_input:
                continue
            
            # Executar o agente
            agent_logger.info("🤖 Agente: Processando...")
            
            result = agent.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
            
            # Extrair resposta final
            final_message = result["messages"][-1]
            response = final_message.content
            
            print(f"🤖 Agente: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Tchau!")
            break
        except Exception as e:
            print(f"❌ Erro: {str(e)}\n")

if __name__ == "__main__":
    # Verificar se o provedor LLM está funcionando
    try:
        test_response = llm.invoke("Hello")
        agent_logger.info("✅ Conexão com provedor LLM estabelecida")
    except Exception as e:
        agent_logger.error(f"❌ Erro ao conectar com provedor LLM: {e}")
        if provider_info['provider'] == 'ollama':
            agent_logger.error("Certifique-se de que o Ollama está rodando com: ollama serve")
        elif provider_info['provider'] == 'openai':
            agent_logger.error("Verifique se a OPENAI_API_KEY está configurada corretamente")
        elif provider_info['provider'] == 'gemini':
            agent_logger.error("Verifique se a GEMINI_API_KEY está configurada corretamente")
        exit(1)
    
    # Executar o agente
    run_agent()