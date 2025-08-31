# mcp_http_server.py

from fastmcp import FastMCP

# Cria uma instância do servidor FastMCP
mcp = FastMCP("Meu Servidor HTTP")

# Define uma ferramenta simples que retorna uma saudação
@mcp.tool
def hello(name: str) -> str:
    """Retorna uma saudação personalizada."""
    return f"Olá, {name}!"

# Define outra ferramenta que soma dois números
@mcp.tool
def add(a: int, b: int) -> int:
    """Adiciona dois números e retorna o resultado."""
    return a + b
