import os
from dotenv import load_dotenv

from tools import tools 
from mcp_bridge import register_langchain_tools_as_mcp

# Carregar variáveis de ambiente
load_dotenv()

# Tentar usar MCP oficial primeiro (para mcp dev), senão FastMCP (para fastmcp run)
try:
    from mcp.server.fastmcp.server import FastMCP as OfficialFastMCP
    mcp = OfficialFastMCP(os.getenv("MCP_SERVER_NAME", "LangGraphToolsMCP"))
    using_official = True
except ImportError:
    from fastmcp import FastMCP
    mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "LangGraphToolsMCP"))
    using_official = False

# Opcional: healthcheck simples
@mcp.tool("ping")
async def ping() -> str:
    """Verifica se o servidor MCP está no ar."""
    return "pong"

# Registra TODAS as suas tools do LangChain como tools MCP
register_langchain_tools_as_mcp(mcp, tools)

if __name__ == "__main__":
    # Sempre usar a implementação simples para ambos os casos
    mcp.run()