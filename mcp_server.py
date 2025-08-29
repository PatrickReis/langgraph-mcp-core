"""
Servidor MCP que expõe suas LangChain Tools (incluindo search_knowledge_base)
como tools MCP.
"""

import os
import asyncio
from mcp.server.fastmcp import FastMCP

# 💡 importe suas definições existentes:
# - embeddings, vectorstore, @tool search_knowledge_base, lista "tools", etc.
from main import tools  # <-- troque pelo nome do seu arquivo

from mcp_bridge import register_langchain_tools_as_mcp


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("LangGraphToolsMCP")

    # Opcional: healthcheck simples
    @mcp.tool("ping")
    async def ping() -> str:
        """Verifica se o servidor MCP está no ar."""
        return "pong"

    # Registra TODAS as suas tools do LangChain como tools MCP
    register_langchain_tools_as_mcp(mcp, tools)

    return mcp


async def run_http_server():
    """Executa o servidor MCP via HTTP"""
    mcp = create_mcp_server()
    await mcp.run_streamable_http_async(host="127.0.0.1", port=8088)

async def run_sse_server():
    """Executa o servidor MCP via Server-Sent Events"""
    mcp = create_mcp_server()
    await mcp.run_sse_async(host="127.0.0.1", port=8088)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        # Executa via HTTP streamable
        asyncio.run(run_http_server())
    elif len(sys.argv) > 1 and sys.argv[1] == "sse":
        # Executa via Server-Sent Events
        asyncio.run(run_sse_server())
    else:
        # Executa via stdio (padrão para clientes MCP)
        mcp = create_mcp_server()
        mcp.run()
