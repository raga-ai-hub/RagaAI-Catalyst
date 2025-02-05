import sys
from typing import Any, Dict, List, Optional

# sys.path.append("/Users/ragaai_user/work/ragaai-catalyst")

from utils.llm import get_llm_response
from .base_agent import BaseAgent
from ragaai_catalyst import trace_agent

class SynthesisAgent(BaseAgent):
    """Agent responsible for synthesizing findings and generate insights."""
    
    @trace_agent('Process Synthesis')
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the findings and generate insights.
        
        Args:
            input_data: Dictionary containing findings and research question
            
        Returns:
            Dictionary containing synthesized results
        """
        findings = input_data.get("findings", [])
        research_question = input_data.get("research_question")
        
        if not findings:
            return {
                "status": "error",
                "message": "No findings provided for synthesis"
            }
        
        # Step 1: Identify patterns and themes
        patterns = await self._identify_patterns(findings)
        
        # Step 2: Generate hypotheses
        hypotheses = await self._generate_hypotheses(patterns, research_question)
        
        # Step 3: Evaluate hypotheses
        evaluated_hypotheses = await self._evaluate_hypotheses(hypotheses, findings)
        
        # Step 4: Generate conclusions
        conclusions = await self._generate_conclusions(evaluated_hypotheses)
        
        # Step 5: Generate recommendations
        recommendations = await self._generate_recommendations(conclusions, research_question)
        
        return {
            "status": "success",
            "patterns": patterns,
            "hypotheses": evaluated_hypotheses,
            "conclusions": conclusions,
            "recommendations": recommendations
        }
    
    async def _identify_patterns(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns and themes in the findings.
        
        Args:
            findings: List of research findings
            
        Returns:
            List of identified patterns
        """
        prompt = """
        Analyze these research findings and identify key patterns and themes:
        
        Findings:
        {}
        
        Identify:
        1. Common themes
        2. Contradictions
        3. Knowledge gaps
        4. Emerging trends
        """.format(self._format_findings_for_prompt(findings))

        model = self.config.get("model", "gpt-4o-mini")
        provider = self.config.get("provider", "openai")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1000)
        async_llm = self.config.get("async_llm", False)
        syntax = self.config.get("syntax", "completion")
        response = await get_llm_response(prompt, model=model, provider=provider, temperature=temperature, max_tokens=max_tokens, async_llm=async_llm, syntax=syntax)
        # Process response to extract patterns
        return [{"type": "pattern", "description": response}]
    
    async def _generate_hypotheses(self, patterns: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Generate hypotheses based on identified patterns.
        
        Args:
            patterns: List of identified patterns
            research_question: Original research question
            
        Returns:
            List of generated hypotheses
        """
        prompt = f"""
        Based on these patterns and the research question:
        
        Research Question: {research_question}
        
        Patterns:
        {self._format_patterns_for_prompt(patterns)}
        
        Generate 3-5 hypotheses that could explain the patterns or answer the research question.
        """
        
        model = self.config.get("model", "gpt-4o-mini")
        provider = self.config.get("provider", "openai")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1000)
        async_llm = self.config.get("async_llm", False)
        syntax = self.config.get("syntax", "completion")
        response = await get_llm_response(prompt, model=model, provider=provider, temperature=temperature, max_tokens=max_tokens, async_llm=async_llm, syntax=syntax)
        # Process response to extract hypotheses
        return [{"hypothesis": response, "confidence": 0.0}]
    
    async def _evaluate_hypotheses(self, hypotheses: List[Dict[str, Any]], findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate generated hypotheses against findings.
        
        Args:
            hypotheses: List of hypotheses to evaluate
            findings: Original research findings
            
        Returns:
            List of evaluated hypotheses with confidence scores
        """
        evaluated_hypotheses = []
        for hypothesis in hypotheses:
            prompt = f"""
            Evaluate this hypothesis against the research findings:
            
            Hypothesis: {hypothesis['hypothesis']}
            
            Findings:
            {self._format_findings_for_prompt(findings)}
            
            Provide:
            1. Supporting evidence
            2. Contradicting evidence
            3. Confidence score (0-1)
            """
            
            model = self.config.get("model", "gpt-4o-mini")
            provider = self.config.get("provider", "openai")
            temperature = self.config.get("temperature", 0.7)
            max_tokens = self.config.get("max_tokens", 1000)
            async_llm = self.config.get("async_llm", False)
            syntax = self.config.get("syntax", "completion")
            response = await get_llm_response(prompt, model=model, provider=provider, temperature=temperature, max_tokens=max_tokens, async_llm=async_llm, syntax=syntax)
            # Process response to extract evaluation
            evaluated_hypotheses.append({
                **hypothesis,
                "evaluation": response,
                "confidence": 0.8  # Placeholder confidence score
            })
        
        return evaluated_hypotheses
    
    async def _generate_conclusions(self, evaluated_hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate conclusions based on evaluated hypotheses.
        
        Args:
            evaluated_hypotheses: List of evaluated hypotheses
            
        Returns:
            List of conclusions
        """
        prompt = """
        Based on these evaluated hypotheses:
        {}
        
        Generate key conclusions that:
        1. Synthesize the most supported hypotheses
        2. Address any contradictions
        3. Highlight remaining uncertainties
        """.format(self._format_hypotheses_for_prompt(evaluated_hypotheses))
        
        model = self.config.get("model", "gpt-4o-mini")
        provider = self.config.get("provider", "openai")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1000)
        async_llm = self.config.get("async_llm", False)
        syntax = self.config.get("syntax", "completion")
        response = await get_llm_response(prompt, model=model, provider=provider, temperature=temperature, max_tokens=max_tokens, async_llm=async_llm, syntax=syntax)
        # Process response to extract conclusions
        return [{"conclusion": response, "confidence": 0.9}]
    
    async def _generate_recommendations(self, conclusions: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Generate recommendations based on conclusions.
        
        Args:
            conclusions: List of research conclusions
            research_question: Original research question
            
        Returns:
            List of recommendations
        """
        prompt = f"""
        Based on these conclusions and the original research question:
        
        Research Question: {research_question}
        
        Conclusions:
        {self._format_conclusions_for_prompt(conclusions)}
        
        Generate:
        1. Key recommendations
        2. Suggested next steps
        3. Areas for further research
        """
        
        model = self.config.get("model", "gpt-4o-mini")
        provider = self.config.get("provider", "openai")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1000)
        async_llm = self.config.get("async_llm", False)
        syntax = self.config.get("syntax", "completion")
        response = await get_llm_response(prompt, model=model, provider=provider, temperature=temperature, max_tokens=max_tokens, async_llm=async_llm, syntax=syntax)
        # Process response to extract recommendations
        return [{"recommendation": response, "priority": "high"}]
    
    def _format_findings_for_prompt(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings for use in LLM prompts."""
        return "\n".join(f"- {finding.get('summary', '')}" for finding in findings)
    
    def _format_patterns_for_prompt(self, patterns: List[Dict[str, Any]]) -> str:
        """Format patterns for use in LLM prompts."""
        return "\n".join(f"- {pattern.get('description', '')}" for pattern in patterns)
    
    def _format_hypotheses_for_prompt(self, hypotheses: List[Dict[str, Any]]) -> str:
        """Format hypotheses for use in LLM prompts."""
        return "\n".join(
            f"- Hypothesis: {h.get('hypothesis', '')}\n  Confidence: {h.get('confidence', 0)}\n  Evaluation: {h.get('evaluation', '')}"
            for h in hypotheses
        )
    
    def _format_conclusions_for_prompt(self, conclusions: List[Dict[str, Any]]) -> str:
        """Format conclusions for use in LLM prompts."""
        return "\n".join(f"- {conclusion.get('conclusion', '')}" for conclusion in conclusions)
