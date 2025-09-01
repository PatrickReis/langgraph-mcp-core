"""
Centralized Application Settings
Professional configuration management for Agent core.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMSettings:
    """LLM provider settings."""
    main_provider: str = "ollama"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:latest"
    ollama_embeddings_model: str = "nomic-embed-text"
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Gemini settings  
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    
    @classmethod
    def from_env(cls) -> 'LLMSettings':
        """Create settings from environment variables."""
        return cls(
            main_provider=os.getenv("MAIN_PROVIDER", "ollama"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3:latest"),
            ollama_embeddings_model=os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        )


@dataclass
class VectorStoreSettings:
    """Vector store settings."""
    persist_directory: str = "./data/vector_stores"
    k_results: int = 3
    collection_name: str = "ai_accelerator_knowledge"
    
    @classmethod
    def from_env(cls) -> 'VectorStoreSettings':
        """Create settings from environment variables."""
        return cls(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_stores"),
            k_results=int(os.getenv("VECTOR_SEARCH_K_RESULTS", "3")),
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", "ai_accelerator_knowledge")
        )


@dataclass
class MCPSettings:
    """MCP server settings."""
    server_name: str = "AIAcceleratorMCP"
    host: str = "127.0.0.1"
    port: int = 8088
    transport: str = "sse"
    
    @classmethod
    def from_env(cls) -> 'MCPSettings':
        """Create settings from environment variables."""
        return cls(
            server_name=os.getenv("MCP_SERVER_NAME", "AIAcceleratorMCP"),
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("MCP_PORT", "8088")),
            transport=os.getenv("MCP_TRANSPORT", "sse")
        )


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    level: str = "INFO"
    log_dir: str = "./logs"
    max_file_size: str = "10MB"
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    
    @classmethod
    def from_env(cls) -> 'LoggingSettings':
        """Create settings from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_dir=os.getenv("LOG_DIR", "./logs"),
            max_file_size=os.getenv("LOG_MAX_FILE_SIZE", "10MB"),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            enable_console=os.getenv("LOG_ENABLE_CONSOLE", "true").lower() == "true",
            enable_file=os.getenv("LOG_ENABLE_FILE", "true").lower() == "true"
        )


@dataclass
class AppSettings:
    """Main application settings."""
    # Component settings (no defaults - will be set in from_env)
    llm: LLMSettings
    vector_store: VectorStoreSettings
    mcp: MCPSettings
    logging: LoggingSettings
    
    # App settings with defaults
    app_name: str = "Agent core"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> 'AppSettings':
        """Create complete settings from environment variables."""
        return cls(
            llm=LLMSettings.from_env(),
            vector_store=VectorStoreSettings.from_env(),
            mcp=MCPSettings.from_env(),
            logging=LoggingSettings.from_env(),
            app_name=os.getenv("APP_NAME", "Agent core"),
            version=os.getenv("APP_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def validate(self) -> bool:
        """Validate settings."""
        # Ensure data directories exist
        Path(self.vector_store.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate LLM settings based on main provider
        if self.llm.main_provider == "openai" and not self.llm.openai_api_key:
            raise ValueError("OPENAI_API_KEY required when using OpenAI provider")
        
        if self.llm.main_provider == "gemini" and not self.llm.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required when using Gemini provider")
        
        return True


# Global settings instance
settings = AppSettings.from_env()
settings.validate()