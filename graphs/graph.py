from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from tools.tools import tools
from logger.logger import get_logger

# Get specialized logger
graph_logger = get_logger("graph")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

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

def call_model(state: AgentState, llm) -> AgentState:
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
            graph_logger.info("Agente decidiu usar a base de conhecimento")
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
            graph_logger.info("💭 Agente respondendo diretamente...")
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

def call_tool(state: AgentState, llm) -> AgentState:
    """Executa as ferramentas via ToolNode com visibilidade"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Lista de ferramentas disponíveis
    tool_node = ToolNode(tools)
    
    # Mostrar quais ferramentas estão sendo executadas
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "unknown")
            graph_logger.info(f"🛠️ Executando ferramenta: {tool_name}")
    
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
        graph_logger.info("📚 Resultado obtido da base de conhecimento")
        
        # Gerar resposta final com base no resultado
        final_prompt = f"""Com base nas informações da base de conhecimento:
{tool_result}

Pergunta original: {messages[0].content if messages else ""}

Sempre responda em pt-br.Forneça uma resposta clara e informativa, usando as informações encontradas."""

        try:
            final_response = llm.invoke(final_prompt)
            result["messages"].append(AIMessage(content=final_response))
            graph_logger.info("Resposta final gerada com base na busca")
        except Exception as e:
            result["messages"].append(AIMessage(content=f"Erro ao gerar resposta final: {e}"))
    
    return result

def create_agent_graph(llm):
    """Cria e configura o grafo do agente"""
    
    # Criar o grafo
    workflow = StateGraph(AgentState)
    
    # Criar funções curried com o llm
    def agent_node(state):
        return call_model(state, llm)
    
    def action_node(state):
        return call_tool(state, llm)
    
    # Adicionar nós
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", action_node)
    
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