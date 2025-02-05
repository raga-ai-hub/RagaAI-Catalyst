
from typing import Any, Dict, Optional

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base agent.
        
        Args:
            config: Optional configuration dictionary for the agent
        """
        self.config = config or {}
        self.memory: Dict[str, Any] = {}

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data and return results.
        
        Args:
            input_data: Dictionary containing input data for processing
            
        Returns:
            Dictionary containing processed results
        """
        pass
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update agent's memory with new information.
        
        Args:
            key: Key to store the information
            value: Value to store
        """
        self.memory[key] = value
    
    def get_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent's memory.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Stored value if exists, None otherwise
        """
        return self.memory.get(key)
