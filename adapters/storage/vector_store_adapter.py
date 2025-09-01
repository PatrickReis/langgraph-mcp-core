"""
Vector Store Adapter
Handles vector database operations using ChromaDB.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.interfaces.storage_repository import StorageRepositoryInterface
from infrastructure.config.settings import settings
from shared.utils.logger import get_logger

storage_logger = get_logger("storage")


class ChromaVectorStoreAdapter(StorageRepositoryInterface):
    """ChromaDB vector store adapter."""
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self._vector_store: Optional[Chroma] = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store."""
        try:
            # Ensure persist directory exists
            persist_dir = Path(settings.vector_store.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            storage_logger.info(f"Initializing ChromaDB at {persist_dir}")
            
            self._vector_store = Chroma(
                collection_name=settings.vector_store.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir)
            )
            
            storage_logger.success("ChromaDB initialized successfully")
            
        except Exception as e:
            storage_logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def store_documents(self, documents: List[Document]) -> bool:
        """Store documents in vector database."""
        try:
            if not self._vector_store:
                storage_logger.error("Vector store não inicializado")
                return False
            
            storage_logger.info(f"Storing {len(documents)} documents")
            self._vector_store.add_documents(documents)
            
            storage_logger.success(f"Stored {len(documents)} documents successfully")
            return True
            
        except Exception as e:
            storage_logger.error(f"Failed to store documents: {str(e)}")
            return False
    
    def search_similar(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents."""
        try:
            if not self._vector_store:
                storage_logger.error("Vector store não inicializado")
                return []
            
            k = k or settings.vector_store.k_results
            storage_logger.info(f"Searching for similar documents: {query[:50]}")
            
            results = self._vector_store.similarity_search(query, k=k)
            storage_logger.success(f"Found {len(results)} similar documents")
            
            return results
            
        except Exception as e:
            storage_logger.error(f"Search failed: {str(e)}")
            return []
    
    def search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Search for similar documents with scores."""
        try:
            if not self._vector_store:
                storage_logger.error("Vector store não inicializado")
                return []
            
            k = k or settings.vector_store.k_results
            storage_logger.info(f"Searching with scores: {query[:50]}")
            
            results = self._vector_store.similarity_search_with_score(query, k=k)
            storage_logger.success(f"Found {len(results)} documents with scores")
            
            return results
            
        except Exception as e:
            storage_logger.error(f"Search with score failed: {str(e)}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            if not self._vector_store:
                storage_logger.error("Vector store não inicializado")
                return False
            
            storage_logger.info("Deleting collection")
            self._vector_store.delete_collection()
            
            storage_logger.success("Collection deleted successfully")
            return True
            
        except Exception as e:
            storage_logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            if not self._vector_store:
                return {"error": "Vector store not initialized"}
            
            # Get collection stats (ChromaDB specific)
            collection = self._vector_store._collection
            count = collection.count()
            
            info = {
                "collection_name": settings.vector_store.collection_name,
                "document_count": count,
                "persist_directory": settings.vector_store.persist_directory
            }
            
            storage_logger.info(f"Collection info retrieved: {count} documents")
            return info
            
        except Exception as e:
            storage_logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}
    
    def get_vector_store(self) -> Optional[Chroma]:
        """Get the underlying vector store instance."""
        return self._vector_store