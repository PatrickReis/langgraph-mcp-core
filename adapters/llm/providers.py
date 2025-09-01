"""
LLM Provider Adapters
Refactored version of llm_providers.py with Clean Architecture.
"""

import os
from typing import Dict, Any
from shared.utils.logger import get_logger

llm_logger = get_logger("llm")
from langchain_core.embeddings import Embeddings

from core.interfaces.llm_provider import LLMProviderInterface
from core.entities.agent import ProviderType


class OllamaProvider(LLMProviderInterface):
    """Ollama LLM Provider Adapter."""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3:latest")
        self.embeddings_model = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text")
        self._llm = None
        self._embeddings = None
    
    def _get_llm(self):
        """Lazy load LLM instance."""
        if self._llm is None:
            try:
                from langchain_ollama import OllamaLLM
                self._llm = OllamaLLM(
                    model=self.model,
                    base_url=self.base_url
                )
                llm_logger.success("Ollama LLM inicializado", model=self.model)
            except ImportError:
                raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")
        return self._llm
    
    def _get_embeddings(self):
        """Lazy load embeddings instance."""
        if self._embeddings is None:
            try:
                from langchain_ollama import OllamaEmbeddings
                self._embeddings = OllamaEmbeddings(
                    model=self.embeddings_model,
                    base_url=self.base_url
                )
                llm_logger.success("Ollama embeddings inicializado", model=self.embeddings_model)
            except ImportError:
                raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")
        return self._embeddings
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama."""
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            llm_logger.error(f"Geração Ollama falhou: {e}")
            raise
    
    def get_provider_type(self) -> ProviderType:
        """Get provider type."""
        return ProviderType.OLLAMA
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "embeddings_model": self.embeddings_model
        }
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_connection(self) -> bool:
        """Test connection to Ollama."""
        try:
            llm = self._get_llm()
            llm.invoke("Hello")
            return True
        except Exception as e:
            llm_logger.error(f"Teste de conexão Ollama falhou: {e}")
            return False


class OpenAIProvider(LLMProviderInterface):
    """OpenAI LLM Provider Adapter."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self._llm = None
        self._embeddings = None
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured in .env file")
    
    def _get_llm(self):
        """Lazy load LLM instance."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key
                )
                llm_logger.success("OpenAI LLM inicializado", model=self.model)
            except ImportError:
                raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        return self._llm
    
    def _get_embeddings(self):
        """Lazy load embeddings instance."""
        if self._embeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings(api_key=self.api_key)
                llm_logger.success("OpenAI embeddings inicializado")
            except ImportError:
                raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        return self._embeddings
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI."""
        try:
            llm = self._get_llm()
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            llm_logger.error(f"Geração OpenAI falhou: {e}")
            raise
    
    def get_provider_type(self) -> ProviderType:
        """Get provider type."""
        return ProviderType.OPENAI
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "api_key_configured": bool(self.api_key)
        }
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.api_key)
    
    def test_connection(self) -> bool:
        """Test connection to OpenAI."""
        try:
            llm = self._get_llm()
            from langchain_core.messages import HumanMessage
            llm.invoke([HumanMessage(content="Hello")])
            return True
        except Exception as e:
            llm_logger.error(f"Teste de conexão OpenAI falhou: {e}")
            return False


class GeminiProvider(LLMProviderInterface):
    """Google Gemini LLM Provider Adapter."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._llm = None
        self._embeddings = None
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured in .env file")
    
    def _get_llm(self):
        """Lazy load LLM instance."""
        if self._llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    google_api_key=self.api_key
                )
                llm_logger.success("Gemini LLM inicializado", model=self.model)
            except ImportError:
                raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
        return self._llm
    
    def _get_embeddings(self):
        """Lazy load embeddings instance.""" 
        if self._embeddings is None:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.api_key)
                llm_logger.success("Gemini embeddings inicializado")
            except ImportError:
                raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
        return self._embeddings
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini."""
        try:
            llm = self._get_llm()
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            llm_logger.error(f"Geração Gemini falhou: {e}")
            raise
    
    def get_provider_type(self) -> ProviderType:
        """Get provider type."""
        return ProviderType.GEMINI
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "gemini",
            "model": self.model,
            "api_key_configured": bool(self.api_key)
        }
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return bool(self.api_key)
    
    def test_connection(self) -> bool:
        """Test connection to Gemini."""
        try:
            llm = self._get_llm()
            from langchain_core.messages import HumanMessage
            llm.invoke([HumanMessage(content="Hello")])
            return True
        except Exception as e:
            llm_logger.error(f"Teste de conexão Gemini falhou: {e}")
            return False


def create_embeddings_provider(provider_type: ProviderType) -> Embeddings:
    """Create embeddings provider based on type."""
    try:
        llm_logger.info(f"Criando provedor de embeddings: {provider_type.value}")
        
        if provider_type == ProviderType.OLLAMA:
            provider = OllamaProvider()
            return provider._get_embeddings()
        elif provider_type == ProviderType.OPENAI:
            provider = OpenAIProvider()
            return provider._get_embeddings()
        elif provider_type == ProviderType.GEMINI:
            provider = GeminiProvider()
            return provider._get_embeddings()
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
            
    except Exception as e:
        llm_logger.error(f"Falha ao criar provedor de embeddings: {e}")
        raise