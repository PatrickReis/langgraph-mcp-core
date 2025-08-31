import os
import asyncio
from fastmcp import FastMCP
from dotenv import load_dotenv

from tools import tools 

from mcp_bridge import register_langchain_tools_as_mcp

# Carregar variáveis de ambiente
load_dotenv()

def create_mcp_server() -> FastMCP:
    mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "LangGraphToolsMCP"))

    # Opcional: healthcheck simples
    @mcp.tool("ping")
    async def ping() -> str:
        """Verifica se o servidor MCP está no ar."""
        return "pong"

    # Registra TODAS as suas tools do LangChain como tools MCP
    register_langchain_tools_as_mcp(mcp, tools)

    return mcp

mcp = create_mcp_server()