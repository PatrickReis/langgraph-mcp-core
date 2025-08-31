import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

print("🔧 Carregando servidor MCP...")

# Tentar usar MCP oficial primeiro (para mcp dev), senão FastMCP (para fastmcp run)
try:
    from mcp.server.fastmcp.server import FastMCP as OfficialFastMCP
    mcp = OfficialFastMCP(os.getenv("MCP_SERVER_NAME", "LangGraphToolsMCP"))
    using_official = True
    print("✅ Usando MCP oficial")
except ImportError:
    from fastmcp import FastMCP
    mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "LangGraphToolsMCP"))
    using_official = False
    print("✅ Usando FastMCP")

# Tool de healthcheck simples
@mcp.tool("ping")
async def ping() -> str:
    """Verifica se o servidor MCP está no ar."""
    return "pong"

print("✅ Tool ping registrada")

# Carregar tools do LangChain
print("📦 Carregando tools...")
try:
    from tools import tools
    print(f"✅ {len(tools)} tools carregadas:")
    for tool in tools:
        print(f"  - {tool.name}")
except Exception as e:
    print(f"❌ Erro ao carregar tools: {e}")
    tools = []

# Carregar e executar bridge
print("🌉 Executando bridge...")
try:
    from mcp_bridge import register_langchain_tools_as_mcp
    register_langchain_tools_as_mcp(mcp, tools)
    print("✅ Bridge concluído com sucesso")
except Exception as e:
    print(f"❌ Erro no bridge: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Iniciando servidor MCP...")
    try:
        mcp.run()
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")
        import traceback
        traceback.print_exc()