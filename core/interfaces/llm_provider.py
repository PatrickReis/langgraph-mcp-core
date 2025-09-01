"""
LLM Provider Interface
Defines contract for LLM provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..entities.agent import ProviderType


class LLMProviderInterface(ABC):
    """Interface for LLM provider implementations."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to provider."""
        pass