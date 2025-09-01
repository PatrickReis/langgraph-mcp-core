"""
Agent Domain Entity
Defines the core Agent business object and configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    PROCESSING = "processing"
    USING_TOOLS = "using_tools"
    ERROR = "error"
    COMPLETED = "completed"


class ProviderType(Enum):
    """LLM Provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class AgentConfig:
    """Agent configuration entity."""
    name: str
    provider: ProviderType
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt
        }


@dataclass
class AgentState:
    """Current agent state entity."""
    status: AgentStatus
    current_task: Optional[str]
    tools_used: List[str]
    last_response: Optional[str]
    error_message: Optional[str] = None
    
    def update_status(self, status: AgentStatus, task: Optional[str] = None):
        """Update agent status."""
        self.status = status
        if task:
            self.current_task = task
    
    def add_tool_usage(self, tool_name: str):
        """Record tool usage."""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)


@dataclass 
class AgentExecution:
    """Agent execution result entity."""
    query: str
    response: str
    tools_used: List[str]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "query": self.query,
            "response": self.response,
            "tools_used": self.tools_used,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message
        }