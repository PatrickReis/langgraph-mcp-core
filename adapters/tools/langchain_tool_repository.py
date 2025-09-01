"""
LangChain Tool Repository Adapter
Implements tool management using LangChain tools.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from core.interfaces.tool_repository import ToolRepositoryInterface, ToolInfo
from shared.utils.logger import get_logger

tool_logger = get_logger("tools")


class KnowledgeSearchTool(BaseTool):
    """Ferramenta customizada para buscar na base de conhecimento."""
    
    name: str = "search_knowledge_base"
    description: str = "Buscar na base de conhecimento por informações relevantes"
    vector_store: Optional[Chroma] = None
    
    def __init__(self, vector_store: Optional[Chroma] = None):
        super().__init__()
        self.vector_store = vector_store
    
    def _run(self, query: str) -> str:
        """Buscar na base de conhecimento."""
        try:
            if not self.vector_store:
                return "Base de conhecimento não disponível"
            
            tool_logger.info(f"Buscando na base de conhecimento: {query[:50]}")
            results = self.vector_store.similarity_search(query, k=3)
            
            if not results:
                return "Nenhuma informação relevante encontrada na base de conhecimento"
            
            response = "Resultados da base de conhecimento:\n"
            for i, doc in enumerate(results, 1):
                response += f"{i}. {doc.page_content[:200]}...\n"
            
            tool_logger.success(f"Encontrados {len(results)} resultados na base de conhecimento")
            return response
            
        except Exception as e:
            tool_logger.error(f"Erro na busca da base de conhecimento: {str(e)}")
            return f"Erro ao buscar na base de conhecimento: {str(e)}"


class CalculatorTool(BaseTool):
    """Ferramenta de calculadora segura."""
    
    name: str = "calculator"
    description: str = "Realizar cálculos matemáticos básicos de forma segura"
    
    def _run(self, expression: str) -> str:
        """Calcular expressão matemática de forma segura."""
        try:
            # Verificação básica de segurança
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Erro: Caracteres inválidos na expressão"
            
            # Avaliar com segurança
            result = eval(expression)
            tool_logger.info(f"Calculado: {expression} = {result}")
            return f"Resultado: {result}"
            
        except Exception as e:
            tool_logger.error(f"Erro na calculadora: {str(e)}")
            return f"Erro de cálculo: {str(e)}"


class LangChainToolRepository(ToolRepositoryInterface):
    """Repositório de ferramentas usando LangChain tools."""
    
    def __init__(self, vector_store: Optional[Chroma] = None):
        self.vector_store = vector_store
        self._tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Inicializar ferramentas disponíveis."""
        tool_logger.info("Inicializando ferramentas LangChain")
        
        # Inicializar cada ferramenta individualmente para evitar falhas em cascata
        
        # Ferramenta de busca na web
        try:
            self._tools["web_search"] = DuckDuckGoSearchRun()
            tool_logger.debug("Ferramenta web_search inicializada")
        except Exception as e:
            tool_logger.error(f"Erro ao inicializar web_search: {str(e)}")
        
        # Ferramenta de busca na Wikipedia
        try:
            wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            self._tools["wikipedia_search"] = wikipedia
            tool_logger.debug("Ferramenta wikipedia_search inicializada")
        except Exception as e:
            tool_logger.error(f"Erro ao inicializar wikipedia_search: {str(e)}")
        
        # Ferramenta de busca na base de conhecimento
        try:
            knowledge_tool = KnowledgeSearchTool(self.vector_store)
            self._tools["search_knowledge_base"] = knowledge_tool
            tool_logger.debug(f"Ferramenta search_knowledge_base inicializada com vector_store: {self.vector_store is not None}")
        except Exception as e:
            tool_logger.error(f"Erro ao inicializar search_knowledge_base: {str(e)}")
        
        # Ferramenta calculadora
        try:
            self._tools["calculator"] = CalculatorTool()
            tool_logger.debug("Ferramenta calculator inicializada")
        except Exception as e:
            tool_logger.error(f"Erro ao inicializar calculator: {str(e)}")
        
        if self._tools:
            tool_logger.success(f"Inicializadas {len(self._tools)} ferramentas: {list(self._tools.keys())}")
        else:
            tool_logger.error("Nenhuma ferramenta foi inicializada com sucesso")
    
    def get_available_tools(self) -> List[ToolInfo]:
        """Obter lista de ferramentas disponíveis."""
        tools = []
        for name, tool in self._tools.items():
            # Extrair parâmetros da descrição da ferramenta ou usar padrões
            parameters = self._extract_tool_parameters(tool)
            tools.append(ToolInfo(
                name=name,
                description=tool.description,
                parameters=parameters
            ))
        
        tool_logger.info(f"Recuperadas {len(tools)} ferramentas disponíveis")
        return tools
    
    def _extract_tool_parameters(self, tool: BaseTool) -> Dict[str, type]:
        """Extrair informações de parâmetros da ferramenta."""
        # Parâmetros padrão baseados em padrões comuns
        param_mapping = {
            "web_search": {"query": str},
            "wikipedia_search": {"query": str},
            "search_knowledge_base": {"query": str},
            "calculator": {"expression": str}
        }
        
        return param_mapping.get(tool.name, {"input": str})
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Executar uma ferramenta com parâmetros fornecidos."""
        tool_logger.tool_execution(tool_name, parameters=parameters)
        
        if not self.is_tool_available(tool_name):
            error_msg = f"Ferramenta '{tool_name}' não está disponível"
            tool_logger.tool_error(tool_name, "ferramenta não disponível")
            return error_msg
        
        try:
            tool = self._tools[tool_name]
            
            # Extrair o parâmetro principal para a ferramenta
            if tool_name in ["web_search", "wikipedia_search", "search_knowledge_base"]:
                query = parameters.get("query", "")
                result = tool.run(query)
            elif tool_name == "calculator":
                expression = parameters.get("expression", "")
                result = tool.run(expression)
            else:
                # Tratamento genérico de parâmetros
                input_value = parameters.get("input", str(parameters))
                result = tool.run(input_value)
            
            tool_logger.tool_success(tool_name)
            return result
            
        except Exception as e:
            tool_logger.tool_error(tool_name, str(e))
            return f"Erro ao executar ferramenta '{tool_name}': {str(e)}"
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Verificar se uma ferramenta está disponível."""
        available = tool_name in self._tools
        if not available:
            tool_logger.debug(f"Ferramenta '{tool_name}' não encontrada. Ferramentas disponíveis: {list(self._tools.keys())}")
        return available
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Obter informações sobre uma ferramenta específica."""
        if not self.is_tool_available(tool_name):
            return None
        
        tool = self._tools[tool_name]
        parameters = self._extract_tool_parameters(tool)
        
        return ToolInfo(
            name=tool_name,
            description=tool.description,
            parameters=parameters
        )