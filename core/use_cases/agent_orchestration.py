"""
Agent Orchestration Use Case
Main business logic for agent execution and decision making.
This is the refactored version of main.py logic.
"""

import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..entities.agent import AgentConfig, AgentState, AgentStatus, AgentExecution
from ..entities.message import Message, Conversation, MessageRole, MessageType, ToolCall
from ..interfaces.llm_provider import LLMProviderInterface
from ..interfaces.tool_repository import ToolRepositoryInterface
from ..interfaces.agent_interface import AgentInterface


class AgentOrchestrationUseCase(AgentInterface):
    """
    Main use case for agent orchestration.
    Handles decision making, tool selection, and response generation.
    """
    
    def __init__(
        self,
        llm_provider: LLMProviderInterface,
        tool_repository: ToolRepositoryInterface,
        config: AgentConfig
    ):
        self.llm_provider = llm_provider
        self.tool_repository = tool_repository
        self.config = config
        self.state = AgentState(
            status=AgentStatus.IDLE,
            current_task=None,
            tools_used=[],
            last_response=None
        )
        
        # Knowledge keywords for decision making
        self.knowledge_keywords = [
            'python', 'programação', 'langgraph', 'chromadb', 'ollama',
            'machine learning', 'ia', 'inteligência artificial', 'rag',
            'deep learning', 'nlp', 'conceito', 'o que é', 'como funciona'
        ]
    
    def execute_query(self, query: str) -> AgentExecution:
        """
        Execute a user query with intelligent tool selection.
        
        Args:
            query: User query string
            
        Returns:
            AgentExecution result with response and metadata
        """
        start_time = time.time()
        self.state.update_status(AgentStatus.PROCESSING, query)
        
        try:
            # Decide if tools are needed
            needs_tools = self._should_use_tools(query)
            
            if needs_tools:
                response = self._execute_with_tools(query)
            else:
                response = self._execute_direct_response(query)
            
            execution_time = time.time() - start_time
            self.state.update_status(AgentStatus.COMPLETED)
            self.state.last_response = response
            
            return AgentExecution(
                query=query,
                response=response,
                tools_used=list(self.state.tools_used),
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent execution failed: {str(e)}"
            
            self.state.update_status(AgentStatus.ERROR)
            self.state.error_message = error_msg
            
            return AgentExecution(
                query=query,
                response=error_msg,
                tools_used=list(self.state.tools_used),
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
    
    def _should_use_tools(self, query: str) -> bool:
        """
        Intelligent decision making for tool usage.
        
        Args:
            query: User query
            
        Returns:
            True if tools should be used
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.knowledge_keywords)
    
    def _execute_with_tools(self, query: str) -> str:
        """
        Execute query using appropriate tools.
        
        Args:
            query: User query
            
        Returns:
            Generated response based on tool results
        """
        self.state.update_status(AgentStatus.USING_TOOLS)
        
        # Use knowledge search tool
        tool_name = "search_knowledge_base"
        self.state.add_tool_usage(tool_name)
        
        # Execute tool
        tool_result = self.tool_repository.execute_tool(tool_name, {"query": query})
        
        # Generate final response based on tool result
        prompt = self._create_tool_response_prompt(query, tool_result)
        response = self.llm_provider.generate_response(prompt)
        
        return response
    
    def _execute_direct_response(self, query: str) -> str:
        """
        Execute query with direct LLM response (no tools).
        
        Args:
            query: User query
            
        Returns:
            Direct LLM response
        """
        return self.llm_provider.generate_response(query)
    
    def _create_tool_response_prompt(self, original_query: str, tool_result: str) -> str:
        """
        Create prompt for final response generation based on tool results.
        
        Args:
            original_query: Original user query
            tool_result: Result from tool execution
            
        Returns:
            Formatted prompt
        """
        return f"""Based on the knowledge base information:
{tool_result}

Original question: {original_query}

Provide a clear and informative answer using the found information."""
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    def reset_state(self):
        """Reset agent state to initial."""
        self.state = AgentState(
            status=AgentStatus.IDLE,
            current_task=None,
            tools_used=[],
            last_response=None
        )
    
    def process_message(self, message: Message) -> AgentExecution:
        """Process a message and return execution result."""
        return self.execute_query(message.content)
    
    def get_capabilities(self) -> list:
        """Get list of agent capabilities."""
        return [
            "text_generation", 
            "knowledge_base_search", 
            "web_search",
            "weather_information",
            "general_tools_usage"
        ]