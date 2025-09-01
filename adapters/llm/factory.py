"""
LLM Provider Factory
Creates and manages LLM provider instances.
"""

import os
from typing import Optional

from core.interfaces.llm_provider import LLMProviderInterface
from core.entities.agent import ProviderType
from shared.utils.logger import get_logger

llm_logger = get_logger("llm")

from .providers import OllamaProvider, OpenAIProvider, GeminiProvider


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers = {
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.GEMINI: GeminiProvider
    }
    
    @classmethod
    def create_provider(cls, provider_type: Optional[ProviderType] = None) -> LLMProviderInterface:
        """
        Create LLM provider instance.
        
        Args:
            provider_type: Type of provider to create. If None, uses MAIN_PROVIDER env var.
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider type is not supported or not configured properly
        """
        if provider_type is None:
            provider_name = os.getenv("MAIN_PROVIDER", "ollama").lower()
            try:
                provider_type = ProviderType(provider_name)
            except ValueError:
                available = [p.value for p in ProviderType]
                raise ValueError(
                    f"Provider '{provider_name}' not supported. "
                    f"Available providers: {', '.join(available)}"
                )
        
        if provider_type not in cls._providers:
            available = [p.value for p in cls._providers.keys()]
            raise ValueError(
                f"Provider '{provider_type.value}' not supported. "
                f"Available providers: {', '.join(available)}"
            )
        
        try:
            provider_class = cls._providers[provider_type]
            provider = provider_class()
            
            llm_logger.info(
                f"LLM provider created",
                provider=provider_type.value,
                model_info=provider.get_model_info()
            )
            
            return provider
            
        except Exception as e:
            llm_logger.error(f"Failed to create provider '{provider_type.value}': {str(e)}")
            raise ValueError(f"Error initializing provider '{provider_type.value}': {str(e)}")
    
    @classmethod
    def get_available_providers(cls) -> list[ProviderType]:
        """Get list of available provider types."""
        return list(cls._providers.keys())
    
    @classmethod
    def test_all_providers(cls) -> dict[ProviderType, bool]:
        """Test all available providers."""
        results = {}
        for provider_type in cls._providers:
            try:
                provider = cls.create_provider(provider_type)
                results[provider_type] = provider.test_connection()
            except Exception:
                results[provider_type] = False
        return results