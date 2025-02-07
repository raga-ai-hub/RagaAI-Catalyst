import sys
from typing import Any, Dict, List, Optional

# sys.path.append("/Users/ragaai_user/work/ragaai-catalyst")

from .base_agent import BaseAgent
from .discovery import DiscoveryAgent
from .synthesis import SynthesisAgent
from ragaai_catalyst import trace_agent

class CoordinatorAgent(BaseAgent):
    """Coordinator agent that manages the research process."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the coordinator agent.
        
        Args:
            config: Optional configuration dictionary
        """
        if not config:
            config = {
                'model': 'gpt-4o-mini',
                'provider': 'openai', 
                'temperature': 0.7,
                'max_tokens': 1000,
                'syntax': 'completion'
            }
        super().__init__(config)
        self.discovery_agent = DiscoveryAgent(config)
        self.synthesis_agent = SynthesisAgent(config)
        self.research_state = {
            "status": "idle",
            "current_phase": None,
            "findings": [],
            "hypotheses": [],
            "conclusions": []
        }
    
    @trace_agent('Coordinate Research')
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the research request and coordinate between agents.
        
        Args:
            input_data: Dictionary containing research question and parameters
            
        Returns:
            Dictionary containing research results and status
        """
        research_question = input_data.get("research_question")
        if not research_question:
            raise ValueError("Research question is required")
        
        # Update research state
        self.research_state["status"] = "active"
        self.research_state["current_phase"] = "discovery"
        
        # Step 1: Delegate to Discovery Agent
        print("Step 1: Delegate to Discovery Agent")
        discovery_results = await self.discovery_agent.process({
            "research_question": research_question,
            "parameters": input_data.get("parameters", {})
        })
        
        self.research_state["findings"] = discovery_results.get("findings", [])
        
        # Step 2: Delegate to Synthesis Agent
        print("Step 2: Delegate to Synthesis Agent")
        self.research_state["current_phase"] = "synthesis"
        synthesis_results = await self.synthesis_agent.process({
            "findings": discovery_results.get("findings", []),
            "research_question": research_question
        })
        
        # Update final research state
        print("Step 3: Update final research state")
        self.research_state.update({
            "status": "completed",
            "current_phase": "completed",
            "hypotheses": synthesis_results.get("hypotheses", []),
            "conclusions": synthesis_results.get("conclusions", [])
        })
        
        return {
            "status": "success",
            "research_state": self.research_state,
            "findings": discovery_results.get("findings", []),
            "conclusions": synthesis_results.get("conclusions", []),
            "recommendations": synthesis_results.get("recommendations", [])
        }
    
    def get_research_state(self) -> Dict[str, Any]:
        """Get the current state of research.
        
        Returns:
            Dictionary containing current research state
        """
        return self.research_state
