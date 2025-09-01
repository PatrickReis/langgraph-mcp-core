"""
Tool Repository Interface
Defines contract for tool management and execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class ToolInfo:
    """Tool information structure."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters


class ToolRepositoryInterface(ABC):
    """Interface for tool repository implementations."""
    
    @abstractmethod
    def get_available_tools(self) -> List[ToolInfo]:
        """Get list of available tools."""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    @abstractmethod
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        pass
    
    @abstractmethod
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get information about a specific tool."""
        pass