"""
Message Domain Entity
Defines core message types and conversation handling.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime


class MessageRole(Enum):
    """Message roles in conversation."""
    HUMAN = "human"
    ASSISTANT = "assistant" 
    SYSTEM = "system"
    TOOL = "tool"


class MessageType(Enum):
    """Message types."""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    ERROR = "error"


@dataclass
class ToolCall:
    """Tool call information."""
    name: str
    args: Dict[str, Any]
    call_id: str


@dataclass
class Message:
    """Core message entity."""
    role: MessageRole
    content: str
    message_type: MessageType = MessageType.TEXT
    tool_calls: Optional[List[ToolCall]] = None
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "role": self.role.value,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.name,
                    "args": tc.args,
                    "call_id": tc.call_id
                }
                for tc in self.tool_calls
            ]
        
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    def is_tool_call(self) -> bool:
        """Check if message contains tool calls."""
        return self.message_type == MessageType.TOOL_CALL and self.tool_calls is not None


@dataclass
class Conversation:
    """Conversation entity containing multiple messages."""
    messages: List[Message]
    conversation_id: str
    created_at: datetime = None
    
    def __post_init__(self):
        """Set creation time if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def add_message(self, message: Message):
        """Add message to conversation."""
        self.messages.append(message)
    
    def get_last_message(self) -> Optional[Message]:
        """Get the last message in conversation."""
        return self.messages[-1] if self.messages else None
    
    def get_human_messages(self) -> List[Message]:
        """Get all human messages."""
        return [msg for msg in self.messages if msg.role == MessageRole.HUMAN]
    
    def get_assistant_messages(self) -> List[Message]:
        """Get all assistant messages."""
        return [msg for msg in self.messages if msg.role == MessageRole.ASSISTANT]
    
    def to_langchain_format(self) -> List[Dict[str, Any]]:
        """Convert to LangChain message format."""
        return [msg.to_dict() for msg in self.messages]