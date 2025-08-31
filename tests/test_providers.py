#!/usr/bin/env python3
"""
Script de teste para demonstrar a funcionalidade de múltiplos provedores LLM.
"""

import os
from dotenv import load_dotenv
from llm_providers import get_provider_info, LLMFactory

# Carregar variáveis de ambiente
load_dotenv()

def test_provider(provider_name: str):
    """Testa um provedor específico."""
    print(f"\n🔍 Testando provedor: {provider_name}")
    print("=" * 50)
    
    try:
        # Criar provedor
        provider = LLMFactory.create_provider(provider_name)
        print(f"✅ Provedor criado com sucesso")
        
        # Obter informações
        info = get_provider_info(provider_name)
        print(f"📋 Informações: {info}")
        
        # Testar LLM (apenas se não for OpenAI/Gemini sem API key)
        if provider_name in ['openai', 'gemini']:
            api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
            if api_key and api_key != f"your_{provider_name}_api_key_here":
                try:
                    llm = provider.get_llm()
                    print(f"✅ LLM configurado: {llm.__class__.__name__}")
                except Exception as e:
                    print(f"⚠️  LLM: {e}")
            else:
                print(f"⚠️  LLM: API key não configurada para {provider_name}")
        else:
            try:
                llm = provider.get_llm()
                print(f"✅ LLM configurado: {llm.__class__.__name__}")
            except Exception as e:
                print(f"⚠️  LLM: {e}")
        
        # Testar embeddings
        try:
            embeddings = provider.get_embeddings()
            print(f"✅ Embeddings configurado: {embeddings.__class__.__name__}")
        except Exception as e:
            print(f"⚠️  Embeddings: {e}")
            
    except Exception as e:
        print(f"❌ Erro ao testar {provider_name}: {e}")

def main():
    """Função principal."""
    print("🚀 Teste de Provedores LLM")
    print("=" * 50)
    
    # Mostrar configuração atual
    current_provider = os.getenv("MAIN_PROVIDER", "ollama")
    print(f"🎯 Provedor principal configurado: {current_provider}")
    
    # Testar todos os provedores
    providers = ["ollama", "openai", "gemini"]
    
    for provider in providers:
        test_provider(provider)
    
    print("\n" + "=" * 50)
    print("📝 Resumo:")
    print(f"   • Provedor atual: {current_provider}")
    print("   • Para trocar de provedor, edite a variável MAIN_PROVIDER no arquivo .env")
    print("   • Para OpenAI/Gemini, configure as respectivas API keys")
    print("   • Para Ollama, certifique-se de que está rodando (ollama serve)")

if __name__ == "__main__":
    main()
