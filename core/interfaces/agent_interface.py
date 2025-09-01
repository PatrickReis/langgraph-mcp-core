"""
Agent Interface
Defines the contract for agent implementations.
"""

from abc import ABC, abstractmethod
from core.entities.message import Message
from core.entities.agent import AgentExecution


class AgentInterface(ABC):
    """Interface for agent implementations."""
    
    @abstractmethod
    def process_message(self, message: Message) -> AgentExecution:
        """Process a message and return execution result."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> list:
        """Get list of agent capabilities."""
        pass
    
    @abstractmethod
    def reset_state(self) -> None:
        """Reset agent state."""
        pass