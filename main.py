import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import operator
from dotenv import load_dotenv
from llm_providers import get_llm, get_embeddings, get_provider_info

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do estado do agente
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Configuração dos provedores LLM
try:
    llm = get_llm()
    embeddings = get_embeddings()
    provider_info = get_provider_info()
    print(f"✅ Provedor LLM configurado: {provider_info['provider']}")
except Exception as e:
    print(f"❌ Erro ao configurar provedor LLM: {e}")
    print("Verifique as configurações no arquivo .env")
    exit(1)

# Inicialização da base vetorial ChromaDB
def initialize_vectorstore():
    # Documentos de exemplo para popular a base
    documents = [
        "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.",
        "LangGraph é uma biblioteca para construir aplicações com múltiplos agentes usando grafos.",
        "ChromaDB é uma base de dados vetorial open-source otimizada para embeddings.",
        "Ollama permite executar grandes modelos de linguagem localmente em sua máquina.",
        "RAG (Retrieval Augmented Generation) combina recuperação de informações com geração de texto.",
        "Machine Learning é um subcampo da inteligência artificial que se concentra no desenvolvimento de algoritmos.",
        "Deep Learning usa redes neurais artificiais com múltiplas camadas para aprender padrões complexos.",
        "Natural Language Processing (NLP) é uma área da IA que ajuda computadores a entender linguagem humana."
    ]
    
    # Converter strings em documentos
    docs = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) for i, doc in enumerate(documents)]
    
    # Criar a base vetorial
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    )
    
    return vectorstore

# Inicializar a base vetorial
vectorstore = initialize_vectorstore()

# Definição da ferramenta de busca vetorial
@tool
def search_knowledge_base(query: str) -> str:
    """
    Busca informações relevantes na base de conhecimento vetorial.
    Útil quando o usuário faz perguntas sobre conceitos técnicos, programação, IA, etc.
    
    Args:
        query: A consulta ou pergunta do usuário
    
    Returns:
        Informações relevantes encontradas na base de conhecimento
    """
    try:
        # Realizar busca por similaridade
        k_results = int(os.getenv("VECTOR_SEARCH_K_RESULTS", "3"))
        docs = vectorstore.similarity_search(query, k=k_results)
        
        if docs:
            results = []
            for i, doc in enumerate(docs, 1):
                results.append(f"{i}. {doc.page_content}")
            
            return "Informações encontradas na base de conhecimento:\n" + "\n".join(results)
        else:
            return "Nenhuma informação relevante encontrada na base de conhecimento."
    
    except Exception as e:
        return f"Erro ao buscar na base de conhecimento: {str(e)}"

# Lista de ferramentas disponíveis
tools = [search_knowledge_base]
tool_node = ToolNode(tools)

# Função para determinar se deve chamar ferramentas
def should_continue(state: AgentState) -> str:
    """Decide se deve chamar ferramentas ou finalizar"""
    messages = state["messages"]
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # Se é uma mensagem AI com tool_calls, execute as ferramentas
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "call_tool"
    
    # Caso contrário, finalize
    return "end"

# Função do agente principal
def call_model(state: AgentState) -> AgentState:
    """Função principal que chama o modelo LLM e decide se deve usar ferramentas"""
    messages = state["messages"]
    
    try:
        # Obter última mensagem do usuário
        last_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break
        
        if not last_human_msg:
            return {"messages": [AIMessage(content="Não consegui encontrar sua pergunta.")]}
        
        # Palavras-chave que indicam necessidade de busca na base de conhecimento
        knowledge_keywords = [
            'python', 'programação', 'langgraph', 'chromadb', 'ollama',
            'machine learning', 'ia', 'inteligência artificial', 'rag',
            'deep learning', 'nlp', 'conceito', 'o que é', 'como funciona'
        ]
        
        user_lower = last_human_msg.lower()
        needs_search = any(keyword in user_lower for keyword in knowledge_keywords)
        
        if needs_search:
            print("🔍 Agente decidiu usar a base de conhecimento...")
            
            # Criar uma mensagem AI que indica tool call
            tool_call_msg = AIMessage(
                content="Vou buscar informações relevantes na base de conhecimento.",
                tool_calls=[{
                    "name": "search_knowledge_base",
                    "args": {"query": last_human_msg},
                    "id": "search_1"
                }]
            )
            
            return {"messages": [tool_call_msg]}
        else:
            print("💭 Agente respondendo diretamente...")
            
            # Converter mensagens para formato de prompt
            prompt_parts = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(msg, HumanMessage):
                        prompt_parts.append(f"Human: {content}")
                    elif isinstance(msg, AIMessage):
                        prompt_parts.append(f"Assistant: {content}")
                
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            
            # Chamar o modelo
            response = llm.invoke(prompt)
            return {"messages": [AIMessage(content=response)]}
    
    except Exception as e:
        error_response = f"Erro ao chamar o modelo: {str(e)}"
        return {"messages": [AIMessage(content=error_response)]}

# Função para executar ferramentas
def call_tool(state: AgentState) -> AgentState:
    """Executa as ferramentas via ToolNode com visibilidade"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Mostrar quais ferramentas estão sendo executadas
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "unknown")
            print(f"🛠️ Executando ferramenta: {tool_name}")
    
    # Executar ferramentas
    result = tool_node.invoke(state)
    
    # Adicionar resposta final baseada no resultado da ferramenta
    tool_messages = result["messages"]
    
    # Encontrar a mensagem de resultado da ferramenta
    tool_result = None
    for msg in tool_messages:
        if hasattr(msg, 'content') and msg.content:
            tool_result = msg.content
            break
    
    if tool_result:
        print("📚 Resultado obtido da base de conhecimento")
        
        # Gerar resposta final com base no resultado
        final_prompt = f"""Com base nas informações da base de conhecimento:
{tool_result}

Pergunta original: {messages[0].content if messages else ""}

Forneça uma resposta clara e informativa, usando as informações encontradas."""

        try:
            final_response = llm.invoke(final_prompt)
            result["messages"].append(AIMessage(content=final_response))
            print("✅ Resposta final gerada com base na busca")
        except Exception as e:
            result["messages"].append(AIMessage(content=f"Erro ao gerar resposta final: {e}"))
    
    return result

# Construir o grafo do agente
def create_agent():
    """Cria e configura o grafo do agente"""
    
    # Criar o grafo
    workflow = StateGraph(AgentState)
    
    # Adicionar nós
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    
    # Definir ponto de entrada
    workflow.set_entry_point("agent")
    
    # Adicionar arestas condicionais
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "call_tool": "action",
            "end": END,
        }
    )
    
    # Após executar ferramentas, finalizar (não voltar para o agente)
    workflow.add_edge("action", END)
    
    # Compilar o grafo
    app = workflow.compile()
    
    return app

# Função principal para interagir com o agente
def run_agent():
    """Executa o agente em um loop interativo"""
    agent = create_agent()
    
    print("🤖 Agente LangGraph inicializado!")
    print("💾 Base vetorial ChromaDB carregada")
    print(f"🚀 Provedor LLM: {provider_info['provider']}")
    print(f"📝 Modelo: {provider_info.get('model', 'N/A')}")
    print("\nDigite suas perguntas (ou 'quit' para sair):\n")
    
    while True:
        try:
            user_input = input("👤 Você: ").strip()
            
            if user_input.lower() in ['quit', 'sair', 'exit']:
                print("👋 Tchau!")
                break
            
            if not user_input:
                continue
            
            # Executar o agente
            print("🤖 Agente: Processando...")
            
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
        print("✅ Conexão com provedor LLM estabelecida")
    except Exception as e:
        print(f"❌ Erro ao conectar com provedor LLM: {e}")
        if provider_info['provider'] == 'ollama':
            print("Certifique-se de que o Ollama está rodando com: ollama serve")
        elif provider_info['provider'] == 'openai':
            print("Verifique se a OPENAI_API_KEY está configurada corretamente")
        elif provider_info['provider'] == 'gemini':
            print("Verifique se a GEMINI_API_KEY está configurada corretamente")
        exit(1)
    
    # Executar o agente
    run_agent()