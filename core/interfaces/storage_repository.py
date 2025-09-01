"""
Storage Repository Interface
Defines contract for storage operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class StorageRepositoryInterface(ABC):
    """Interface for storage repository operations."""
    
    @abstractmethod
    def store_documents(self, documents: List[Document]) -> bool:
        """Store documents in the storage system."""
        pass
    
    @abstractmethod
    def search_similar(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Search for similar documents with relevance scores."""
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        pass