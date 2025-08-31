#!/usr/bin/env python3
"""
Script de teste para o servidor MCP.
Testa conectividade e funcionalidades básicas.
"""

import json
import requests
import asyncio
from typing import Dict, Any

class MCPTester:
    """Classe para testar o servidor MCP."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8088"):
        self.base_url = base_url
        self.mcp_endpoint = f"{base_url}/mcp"
        self.session_id = None
        self.headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }
    
    def test_connection(self) -> bool:
        """Testa se o servidor está respondendo."""
        try:
            response = requests.get(self.base_url, timeout=5)
            print(f"✅ Servidor respondendo em {self.base_url}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao conectar: {e}")
            return False
    
    def initialize_session(self) -> bool:
        """Inicializa uma sessão MCP."""
        print("\n🔍 Inicializando sessão MCP...")
        
        # Tentar método initialize (padrão MCP)
        result = self.send_mcp_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "MCP-Tester",
                "version": "1.0.0"
            }
        })
        
        if "error" not in result:
            print(f"✅ Sessão inicializada: {result}")
            return True
        else:
            print(f"⚠️  Initialize falhou: {result['error']}")
            
            # Tentar método alternativo
            print("🔄 Tentando método alternativo...")
            result = self.send_mcp_request("ping")
            
            if "error" not in result:
                print(f"✅ Ping funcionou: {result}")
                return True
            else:
                print(f"❌ Ping também falhou: {result['error']}")
                return False
    
    def send_mcp_request(self, method: str, params: Dict[str, Any] = None, request_id: int = 1) -> Dict[str, Any]:
        """Envia uma requisição MCP."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        
        if params:
            payload["params"] = params
        
        try:
            # Headers específicos para MCP
            mcp_headers = {
                "Accept": "text/event-stream, application/json",
                "Content-Type": "application/json",
                "User-Agent": "MCP-Tester/1.0"
            }
            
            # Adicionar session ID se disponível
            if self.session_id:
                mcp_headers["X-MCP-Session-ID"] = self.session_id
            
            response = requests.post(
                self.mcp_endpoint,
                headers=mcp_headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    # Extrair session ID da resposta se disponível
                    if "result" in result and isinstance(result["result"], dict):
                        if "session_id" in result["result"]:
                            self.session_id = result["result"]["session_id"]
                    return result
                except json.JSONDecodeError:
                    # Tentar extrair session ID de resposta text/event-stream
                    response_text = response.text
                    if "session_id" in response_text:
                        import re
                        match = re.search(r'"session_id":"([^"]+)"', response_text)
                        if match:
                            self.session_id = match.group(1)
                    return {"result": response_text}
            else:
                print(f"⚠️  Status HTTP: {response.status_code}")
                print(f"📝 Response: {response.text}")
                return {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def test_ping(self) -> bool:
        """Testa o endpoint ping."""
        print("\n🔍 Testando ping...")
        result = self.send_mcp_request("ping")
        
        if "error" not in result:
            print(f"✅ Ping: {result}")
            return True
        else:
            print(f"❌ Ping falhou: {result['error']}")
            return False
    
    def test_tools_list(self) -> bool:
        """Testa listagem de ferramentas."""
        print("\n🔍 Testando listagem de ferramentas...")
        result = self.send_mcp_request("tools/list")
        
        if "error" not in result:
            print(f"✅ Ferramentas disponíveis: {result}")
            return True
        else:
            print(f"❌ Listagem de ferramentas falhou: {result['error']}")
            return False
    
    def test_tool_call(self, tool_name: str, tool_input: str) -> bool:
        """Testa chamada de uma ferramenta específica."""
        print(f"\n🔍 Testando ferramenta: {tool_name}")
        
        params = {
            "name": tool_name,
            "arguments": {"input": tool_input}
        }
        
        result = self.send_mcp_request("tools/call", params)
        
        if "error" not in result:
            print(f"✅ Resultado da ferramenta: {result}")
            return True
        else:
            print(f"❌ Chamada da ferramenta falhou: {result['error']}")
            return False
    
    def run_all_tests(self):
        """Executa todos os testes."""
        print("🚀 Iniciando testes do servidor MCP")
        print("=" * 50)
        
        # Teste de conexão básica
        if not self.test_connection():
            print("❌ Servidor não está rodando. Inicie com:")
            print("   fastmcp run mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8088")
            return
        
        # Inicializar sessão MCP
        if not self.initialize_session():
            print("❌ Falha ao inicializar sessão MCP")
            return
        
        # Teste MCP
        tests_passed = 0
        total_tests = 3
        
        if self.test_ping():
            tests_passed += 1
        
        if self.test_tools_list():
            tests_passed += 1
        
        # Teste de ferramenta específica (se disponível)
        if self.test_tool_call("search_knowledge_base", "Python"):
            tests_passed += 1
        
        # Resumo
        print("\n" + "=" * 50)
        print(f"📊 Resultados: {tests_passed}/{total_tests} testes passaram")
        
        if tests_passed == total_tests:
            print("🎉 Todos os testes passaram! Servidor MCP funcionando perfeitamente.")
        else:
            print("⚠️  Alguns testes falharam. Verifique as configurações.")
        
        print("\n💡 Para usar o MCP Inspector:")
        print("   1. Instale: pip install mcp-inspector")
        print("   2. Execute: mcp-inspector")
        print("   3. Conecte em: http://127.0.0.1:8088/mcp")

def main():
    """Função principal."""
    tester = MCPTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
