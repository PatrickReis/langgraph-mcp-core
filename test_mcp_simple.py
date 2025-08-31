#!/usr/bin/env python3
"""
Teste simples do MCP server para debug
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Tentar usar MCP oficial primeiro
try:
    from mcp.server.fastmcp.server import FastMCP as OfficialFastMCP
    mcp = OfficialFastMCP("TestMCP")
    using_official = True
    print("✅ Usando MCP oficial")
except ImportError:
    from fastmcp import FastMCP
    mcp = FastMCP("TestMCP")
    using_official = False
    print("✅ Usando FastMCP")

@mcp.tool("test_tool")
async def test_tool(message: str = "hello") -> str:
    """Tool de teste simples"""
    return f"Resposta: {message}"

print("📝 Tool teste registrada")

if __name__ == "__main__":
    print("🚀 Iniciando servidor...")
    mcp.run()