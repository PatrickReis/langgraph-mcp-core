# tools.py
import os
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.tools import tool
from dotenv import load_dotenv
from llm_providers import get_embeddings

# Carregar variáveis de ambiente
load_dotenv()

# Configuração dos embeddings
try:
    embeddings = get_embeddings()
except Exception as e:
    print(f"❌ Erro ao configurar embeddings: {e}")
    raise

# Inicialização da base vetorial ChromaDB
def initialize_vectorstore():
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
    docs = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) for i, doc in enumerate(documents)]
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    )
    return vectorstore

# Inicializar a base vetorial (apenas uma vez)
if not os.path.exists(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")):
    vectorstore = initialize_vectorstore()
else:
    # Carregar base existente
    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
        embedding_function=embeddings
    )

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
        k_results = int(os.getenv("VECTOR_SEARCH_K_RESULTS", "3"))
        docs = vectorstore.similarity_search(query, k=k_results)
        if docs:
            # Remove documentos duplicados
            unique_docs = []
            seen_content = set()
            for doc in docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            results = [f"{i}. {doc.page_content}" for i, doc in enumerate(unique_docs, 1)]
            return "Informações encontradas na base de conhecimento:\n" + "\n".join(results)
        else:
            return "Nenhuma informação relevante encontrada na base de conhecimento."
    except Exception as e:
        return f"Erro ao buscar na base de conhecimento: {str(e)}"
    
@tool
def simple_calculator(expression: str) -> str:
    """
    Calculadora simples que avalia expressões matemáticas.
    
    Args:
        expression: Expressão matemática como string (ex: "2 + 2", "10 * 5")
    
    Returns:
        Resultado da operação matemática
    """
    try:
        # Avaliação segura apenas de operações básicas
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Erro: Apenas operações matemáticas básicas são permitidas"
        
        result = eval(expression)
        return f"Resultado: {result}"
    except Exception as e:
        return f"Erro na operação: {str(e)}"

@tool  
def echo_tool(message: str) -> str:
    """
    Tool de echo que simplesmente retorna a mensagem recebida.
    
    Args:
        message: Mensagem a ser ecoada
    
    Returns:
        A mesma mensagem recebida
    """
    return f"Echo: {message}"


tools = [search_knowledge_base, echo_tool, simple_calculator]
