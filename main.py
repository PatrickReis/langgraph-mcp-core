import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import add_messages
import operator
from dotenv import load_dotenv
from llm_providers import get_llm, get_embeddings, get_provider_info

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do estado do agente
class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    tool_calls: List[Dict[str, Any]]
    final_answer: str

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
tool_executor = ToolExecutor(tools)

# Função para determinar se deve chamar ferramentas
def should_continue(state: AgentState) -> str:
    """Decide se deve chamar ferramentas ou finalizar"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Se há tool_calls na última mensagem, execute as ferramentas
    if "tool_calls" in last_message and last_message["tool_calls"]:
        return "call_tool"
    
    # Caso contrário, finalize
    return "end"

# Função do agente principal
def call_model(state: AgentState) -> AgentState:
    
    """Função principal que chama o modelo LLM"""
    messages = state["messages"]
    
    # Prompt do sistema para orientar o agente
    system_prompt = """Você é um assistente inteligente com acesso a uma base de conhecimento vetorial.

Quando o usuário fizer perguntas sobre:
- Conceitos de programação (Python, frameworks, etc.)
- Inteligência Artificial e Machine Learning
- Tecnologias como LangGraph, ChromaDB, Ollama
- Processamento de linguagem natural
- Qualquer tópico técnico

Você DEVE usar a ferramenta 'search_knowledge_base' para buscar informações relevantes antes de responder.

Para usar uma ferramenta, responda no formato JSON:
{
    "tool_calls": [
        {
            "name": "search_knowledge_base",
            "args": {
                "query": "sua consulta aqui"
            }
        }
    ]
}

Caso contrário, responda normalmente sem usar ferramentas."""
    
    # Preparar mensagens com contexto do sistema
    prompt_messages = [{"role": "system", "content": system_prompt}] + messages
    
    # Criar prompt como string
    prompt_text = ""
    for msg in prompt_messages:
        role = msg["role"]
        content = msg["content"]
        prompt_text += f"{role}: {content}\n"
    
    prompt_text += "assistant:"
    
    # Chamar o modelo
    try:
        response = llm.invoke(prompt_text)
        
        # Tentar interpretar como JSON (para tool calls)
        try:
            response_json = json.loads(response.strip())
            if "tool_calls" in response_json:
                return {
                    "messages": messages + [{
                        "role": "assistant", 
                        "content": response,
                        "tool_calls": response_json["tool_calls"]
                    }]
                }
        except json.JSONDecodeError:
            pass
        
        # Resposta normal sem tool calls
        return {
            "messages": messages + [{"role": "assistant", "content": response}]
        }
    
    except Exception as e:
        error_response = f"Erro ao chamar o modelo: {str(e)}"
        return {
            "messages": messages + [{"role": "assistant", "content": error_response}]
        }

# Função para executar ferramentas
def call_tool(state: AgentState) -> AgentState:
    """Executa as ferramentas solicitadas"""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = last_message.get("tool_calls", [])
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Executar a ferramenta
        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_args
        )
        
        result = tool_executor.invoke(action)
        results.append(f"Resultado da ferramenta {tool_name}: {result}")
    
    # Adicionar resultados das ferramentas às mensagens
    tool_message = {
        "role": "tool",
        "content": "\n".join(results)
    }
    
    # Gerar resposta final baseada nos resultados
    final_prompt = f"""Com base nos resultados das ferramentas:
{tool_message['content']}

Responda à pergunta do usuário de forma clara e concisa."""
    
    try:
        final_response = llm.invoke(final_prompt)
        
        return {
            "messages": messages + [tool_message, {
                "role": "assistant", 
                "content": final_response
            }]
        }
    except Exception as e:
        error_response = f"Erro ao processar resultado da ferramenta: {str(e)}"
        return {
            "messages": messages + [tool_message, {
                "role": "assistant", 
                "content": error_response
            }]
        }

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
    
    # Após executar ferramentas, voltar para o agente
    workflow.add_edge("action", "agent")
    
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
                "messages": [{"role": "user", "content": user_input}]
            })
            
            # Extrair resposta final
            final_message = result["messages"][-1]
            response = final_message["content"]
            
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