"""
Professional logging configuration for Agent core.
Supports structured logging with different levels and outputs.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class AgentCoreLogger:
    """Professional logger for Agent core project."""
    
    def __init__(self, name: str = "agent_core", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers with proper formatting."""
        
        # Console handler with colored output (using stderr for MCP compatibility)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        
        # File handler for persistent logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"agent_core_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Professional formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, msg: str, **kwargs):
        """Log info message with optional context."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"{msg}{extra}")
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with optional context."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.debug(f"{msg}{extra}")
    
    def warning(self, msg: str, **kwargs):
        """Log warning message with optional context."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.warning(f"{msg}{extra}")
    
    def error(self, msg: str, **kwargs):
        """Log error message with optional context."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.error(f"{msg}{extra}")
    
    def success(self, msg: str, **kwargs):
        """Log success message (info level with ✅)."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"✅ {msg}{extra}")
    
    def progress(self, msg: str, **kwargs):
        """Log progress message (info level with 🔄)."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"🔄 {msg}{extra}")
    
    def agent_decision(self, msg: str, **kwargs):
        """Log agent decision (info level with 🤖)."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"🤖 {msg}{extra}")
    
    def tool_execution(self, tool_name: str, **kwargs):
        """Log tool execution (info level with 🛠️)."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"🛠️ Executing tool: {tool_name}{extra}")
    
    def knowledge_search(self, query: str, **kwargs):
        """Log knowledge base search (info level with 🔍)."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"🔍 Knowledge search: {query[:50]}{extra}")
    
    def tool_error(self, tool_name: str, error: str, **kwargs):
        """Log tool execution error (error level with ❌)."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.error(f"❌ Tool error [{tool_name}]: {error}{extra}")
    
    def tool_success(self, tool_name: str, msg: str = "", **kwargs):
        """Log tool execution success (info level with ✅)."""
        extra = f" | {kwargs}" if kwargs else ""
        message = f"✅ Tool success [{tool_name}]"
        if msg:
            message += f": {msg}"
        self.logger.info(f"{message}{extra}")


# Global logger instances for different components
def get_logger(component: str, level: str = "INFO") -> AgentCoreLogger:
    """Get a logger for a specific component."""
    return AgentCoreLogger(f"agent_core.{component}", level)


# Pre-configured loggers for main components
mcp_logger = get_logger("mcp")
llm_logger = get_logger("llm") 
agent_logger = get_logger("agent")
tool_logger = get_logger("tools")
bridge_logger = get_logger("bridge")

# Main application logger
app_logger = get_logger("app")