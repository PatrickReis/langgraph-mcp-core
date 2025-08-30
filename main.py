import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Union
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import operator

# Importar ferramentas do arquivo tools.py
from tools import tools, search_knowledge_base

# Configuração do estado do agente
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], add_messages]

# Configuração do Ollama
llm = OllamaLLM(
    model="llama3:latest",
    base_url="http://localhost:11434"  # URL padrão do Ollama
)

# Configurar ToolNode com as ferramentas importadas
tool_node = ToolNode(tools)

# Função para determinar se deve chamar ferramentas
def should_continue(state: AgentState) -> str:
    """Decide se deve chamar ferramentas ou finalizar"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Se há tool_calls na última mensagem, execute as ferramentas
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
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
    
    # Criar prompt como string
    prompt_text = f"system: {system_prompt}\n"
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prompt_text += f"user: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            prompt_text += f"assistant: {msg.content}\n"
        elif isinstance(msg, ToolMessage):
            prompt_text += f"tool: {msg.content}\n"
    
    prompt_text += "assistant:"
    
    # Chamar o modelo
    try:
        response = llm.invoke(prompt_text)
        
        # Tentar interpretar como JSON (para tool calls)
        try:
            response_json = json.loads(response.strip())
            if "tool_calls" in response_json:
                ai_message = AIMessage(content=response)
                ai_message.tool_calls = response_json["tool_calls"]
                return {"messages": messages + [ai_message]}
        except json.JSONDecodeError:
            pass
        
        # Resposta normal sem tool calls
        return {"messages": messages + [AIMessage(content=response)]}
    
    except Exception as e:
        error_response = f"Erro ao chamar o modelo: {str(e)}"
        return {"messages": messages + [AIMessage(content=error_response)]}

# Função para executar ferramentas
def call_tool(state: AgentState) -> AgentState:
    """Executa as ferramentas solicitadas"""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = getattr(last_message, 'tool_calls', [])
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Executar a ferramenta usando a função importada
        if tool_name == "search_knowledge_base":
            result = search_knowledge_base.invoke(tool_args)
            results.append(f"Resultado da ferramenta {tool_name}: {result}")
        else:
            results.append(f"Ferramenta {tool_name} não encontrada")
    
    # Adicionar resultados das ferramentas às mensagens
    tool_message = ToolMessage(content="\n".join(results), tool_call_id="1")
    
    # Gerar resposta final baseada nos resultados
    final_prompt = f"""Com base nos resultados das ferramentas:
{tool_message.content}

Responda à pergunta do usuário de forma clara e concisa."""
    
    try:
        final_response = llm.invoke(final_prompt)
        
        return {
            "messages": messages + [tool_message, AIMessage(content=final_response)]
        }
    except Exception as e:
        error_response = f"Erro ao processar resultado da ferramenta: {str(e)}"
        return {
            "messages": messages + [tool_message, AIMessage(content=error_response)]
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
    print("🦙 Usando Ollama com llama3:latest")
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
        except EOFError:
            print("\n👋 Tchau!")
            break
        except Exception as e:
            print(f"❌ Erro: {str(e)}\n")

if __name__ == "__main__":
    # Verificar se o Ollama está executando
    try:
        test_response = llm.invoke("Hello")
        print("✅ Conexão com Ollama estabelecida")
    except Exception as e:
        print(f"❌ Erro ao conectar com Ollama: {e}")
        print("Certifique-se de que o Ollama está rodando com: ollama serve")
        exit(1)
    
    # Executar o agente
    run_agent()