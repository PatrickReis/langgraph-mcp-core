# tools.py
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.tools import tool
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do Ollama para embeddings
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

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

tools = [search_knowledge_base]
