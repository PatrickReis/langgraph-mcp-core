"""
MCP Server Adapter
Handles Model Context Protocol server integration.
"""

from typing import Dict, Any, List, Optional, Callable
import json
from fastmcp import FastMCP

from core.interfaces.agent_interface import AgentInterface
from core.entities.message import Message, MessageType, MessageRole
from shared.utils.logger import get_logger

mcp_logger = get_logger("mcp")


class MCPServerAdapter:
    """MCP server adapter for Clean Architecture."""
    
    def __init__(self, mcp_instance: FastMCP):
        self.mcp = mcp_instance
        self.registered_agents: Dict[str, AgentInterface] = {}
        mcp_logger.info("MCP Server Adapter initialized")
    
    def register_agent_as_tool(
        self, 
        agent: AgentInterface, 
        tool_name: str,
        description: str = "AI Agent tool"
    ):
        """Register an agent as an MCP tool."""
        try:
            mcp_logger.info(f"Registering agent as MCP tool: {tool_name}")
            
            # Store agent reference
            self.registered_agents[tool_name] = agent
            
            # Create MCP tool function
            @self.mcp.tool(description)
            def agent_tool(query: str) -> str:
                """Execute agent query."""
                try:
                    mcp_logger.info(f"Executing agent tool '{tool_name}' with query: {query[:100]}")
                    
                    # Create message
                    message = Message(
                        role=MessageRole.HUMAN,
                        content=query,
                        message_type=MessageType.TEXT,
                        metadata={"source": "mcp_tool", "tool_name": tool_name}
                    )
                    
                    # Execute agent
                    response = agent.process_message(message)
                    
                    if response.success:
                        mcp_logger.success(f"Agent tool '{tool_name}' executed successfully")
                        
                        # Format response with tool usage info
                        result = response.response
                        if response.tools_used:
                            result += f"\n\nTools used: {', '.join(response.tools_used)}"
                        
                        return result
                    else:
                        error_msg = f"Agent execution failed: {response.error_message}"
                        mcp_logger.error(error_msg)
                        return error_msg
                        
                except Exception as e:
                    error_msg = f"Error in agent tool '{tool_name}': {str(e)}"
                    mcp_logger.error(error_msg)
                    return error_msg
            
            # Set function name dynamically
            agent_tool.__name__ = tool_name
            
            mcp_logger.success(f"Agent '{tool_name}' registered as MCP tool")
            
        except Exception as e:
            mcp_logger.error(f"Failed to register agent as tool: {str(e)}")
            raise
    
    def register_function_as_tool(
        self, 
        func: Callable, 
        tool_name: str, 
        description: str
    ):
        """Register a function as an MCP tool."""
        try:
            mcp_logger.info(f"Registering function as MCP tool: {tool_name}")
            
            # Register with MCP
            decorated_func = self.mcp.tool(description)(func)
            decorated_func.__name__ = tool_name
            
            mcp_logger.success(f"Function '{tool_name}' registered as MCP tool")
            
        except Exception as e:
            mcp_logger.error(f"Failed to register function as tool: {str(e)}")
            raise
    
    def get_registered_tools(self) -> List[str]:
        """Get list of registered tool names."""
        # This would depend on FastMCP's API to list tools
        # For now, return registered agents
        return list(self.registered_agents.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered tool."""
        if tool_name in self.registered_agents:
            agent = self.registered_agents[tool_name]
            return {
                "name": tool_name,
                "type": "agent",
                "description": f"AI Agent: {agent.__class__.__name__}"
            }
        return None
    
    def run_server(self, host: str = "127.0.0.1", port: int = 8088):
        """Run the MCP server."""
        try:
            mcp_logger.info(f"Starting MCP server on {host}:{port}")
            self.mcp.run(transport="sse")
            
        except Exception as e:
            mcp_logger.error(f"Failed to run MCP server: {str(e)}")
            raise