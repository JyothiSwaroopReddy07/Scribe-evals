"""Enhanced LLM Judge with confidence scoring, retry mechanisms, and ensemble support."""

import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None


logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for LLM responses."""
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"  # 0.7 - 0.9
    MEDIUM = "medium"  # 0.5 - 0.7
    LOW = "low"  # 0.3 - 0.5
    VERY_LOW = "very_low"  # < 0.3


@dataclass
class UncertaintyMetrics:
    """Quantify uncertainty in LLM responses."""
    confidence_score: float  # 0-1
    confidence_level: ConfidenceLevel
    reasoning_steps: int = 0
    evidence_strength: float = 0.0  # 0-1
    model_agreement: Optional[float] = None  # For ensemble
    entropy: Optional[float] = None  # For probabilistic models
    
    @classmethod
    def from_confidence(cls, confidence: float) -> "UncertaintyMetrics":
        """Create uncertainty metrics from confidence score."""
        if confidence > 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence > 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence > 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence > 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        return cls(
            confidence_score=confidence,
            confidence_level=level
        )


@dataclass
class EnhancedLLMResponse:
    """Enhanced response from LLM judge with confidence and interpretability."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    uncertainty: Optional[UncertaintyMetrics] = None
    reasoning_trace: List[str] = field(default_factory=list)
    raw_logprobs: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    latency_ms: float = 0.0
    fallback_used: bool = False


class EnhancedLLMJudge:
    """
    Enhanced LLM judge with advanced features:
    - Confidence scoring and uncertainty quantification
    - Retry mechanisms with exponential backoff
    - Chain-of-thought reasoning tracking
    - Fallback strategies
    - Support for ensemble evaluation
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_cot: bool = True,
        enable_logprobs: bool = False
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_cot = enable_cot
        self.enable_logprobs = enable_logprobs
        
        # Initialize API client
        if "gpt" in model or "o1" in model:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found")
            self.client = OpenAI(api_key=api_key)
            self.provider = "openai"
        elif "claude" in model:
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.provider = "anthropic"
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None,
        extract_confidence: bool = True
    ) -> EnhancedLLMResponse:
        """
        Call LLM for evaluation with retry and error handling.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query/content to evaluate
            response_format: Optional format specification (e.g., "json")
            extract_confidence: Whether to extract confidence from response
            
        Returns:
            EnhancedLLMResponse object
        """
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        # Add chain-of-thought prompting if enabled
        if self.enable_cot and response_format == "json":
            system_prompt = self._add_cot_instructions(system_prompt)
        
        while retry_count <= self.max_retries:
            try:
                if self.provider == "openai":
                    response = self._call_openai(
                        system_prompt, user_prompt, response_format
                    )
                elif self.provider == "anthropic":
                    response = self._call_anthropic(
                        system_prompt, user_prompt
                    )
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                response.latency_ms = latency_ms
                response.retry_count = retry_count
                
                # Extract confidence and uncertainty metrics
                if extract_confidence and response.content:
                    response.uncertainty = self._extract_uncertainty(response.content)
                
                return response
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count <= self.max_retries:
                    delay = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.warning(
                        f"LLM call failed (attempt {retry_count}/{self.max_retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {self.max_retries} retries: {e}")
        
        # Return fallback response if all retries failed
        return self._create_fallback_response(last_error, retry_count)
    
    def evaluate_ensemble(
        self,
        system_prompt: str,
        user_prompt: str,
        models: List[str],
        response_format: Optional[str] = None
    ) -> List[EnhancedLLMResponse]:
        """
        Evaluate with multiple models for ensemble voting.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query/content to evaluate
            models: List of model names to use
            response_format: Optional format specification
            
        Returns:
            List of EnhancedLLMResponse objects from each model
        """
        responses = []
        
        for model in models:
            try:
                # Create temporary judge with this model
                temp_judge = EnhancedLLMJudge(
                    model=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    max_retries=2,  # Fewer retries for ensemble
                    enable_cot=self.enable_cot
                )
                
                response = temp_judge.evaluate(
                    system_prompt, user_prompt, response_format
                )
                responses.append(response)
                
            except Exception as e:
                logger.warning(f"Failed to get response from {model}: {e}")
                continue
        
        return responses
    
    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None
    ) -> EnhancedLLMResponse:
        """Call OpenAI API with enhanced features."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        
        # Enable logprobs for confidence estimation
        if self.enable_logprobs and "gpt-4" in self.model:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        
        response = self.client.chat.completions.create(**kwargs)
        
        # Extract logprobs if available
        raw_logprobs = None
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            raw_logprobs = {
                "content": [
                    {
                        "token": lp.token,
                        "logprob": lp.logprob,
                        "top_logprobs": [
                            {"token": tlp.token, "logprob": tlp.logprob}
                            for tlp in (lp.top_logprobs or [])
                        ]
                    }
                    for lp in response.choices[0].logprobs.content
                ]
            }
        
        return EnhancedLLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            raw_logprobs=raw_logprobs
        )
    
    def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> EnhancedLLMResponse:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return EnhancedLLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        )
    
    def _add_cot_instructions(self, system_prompt: str) -> str:
        """Add chain-of-thought reasoning instructions."""
        cot_instruction = """

IMPORTANT: Before providing your final JSON response, think through your reasoning step-by-step:
1. First, identify the key evidence from the source material
2. Analyze each piece of evidence systematically
3. Consider alternative interpretations
4. Assess your confidence level based on the strength of evidence
5. Then provide your structured JSON response

Include a "reasoning_steps" field in your JSON response that lists your key reasoning steps."""
        
        return system_prompt + cot_instruction
    
    def _extract_uncertainty(self, content: str) -> UncertaintyMetrics:
        """Extract uncertainty metrics from response content."""
        try:
            # Try to parse JSON response
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0].strip()
            else:
                json_content = content.strip()
            
            data = json.loads(json_content)
            
            # Extract confidence score
            confidence = data.get("confidence", data.get("confidence_score", 0.7))
            
            # Extract reasoning steps
            reasoning_steps = 0
            if "reasoning_steps" in data:
                if isinstance(data["reasoning_steps"], list):
                    reasoning_steps = len(data["reasoning_steps"])
                elif isinstance(data["reasoning_steps"], int):
                    reasoning_steps = data["reasoning_steps"]
            
            # Calculate evidence strength based on number of findings
            evidence_count = 0
            for key in ["hallucinations", "missing_items", "accuracy_issues", "findings"]:
                if key in data and isinstance(data[key], list):
                    evidence_count += len(data[key])
            
            # Normalize evidence strength (0-1 scale, diminishing returns)
            evidence_strength = min(1.0, evidence_count / (evidence_count + 10))
            
            metrics = UncertaintyMetrics.from_confidence(confidence)
            metrics.reasoning_steps = reasoning_steps
            metrics.evidence_strength = evidence_strength
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not extract uncertainty metrics: {e}")
            return UncertaintyMetrics.from_confidence(0.7)
    
    def _create_fallback_response(
        self, 
        error: Exception, 
        retry_count: int
    ) -> EnhancedLLMResponse:
        """Create a fallback response when all retries fail."""
        logger.error(f"Creating fallback response after {retry_count} retries")
        
        fallback_content = json.dumps({
            "error": "LLM evaluation failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "confidence": 0.0,
            "fallback": True
        })
        
        return EnhancedLLMResponse(
            content=fallback_content,
            model=self.model,
            usage=None,
            uncertainty=UncertaintyMetrics.from_confidence(0.0),
            retry_count=retry_count,
            fallback_used=True
        )
    
    def parse_json_response(self, response: EnhancedLLMResponse) -> Dict[str, Any]:
        """Parse JSON response from LLM with enhanced error handling."""
        if response.fallback_used:
            try:
                return json.loads(response.content)
            except:
                return {"error": "Fallback response", "confidence": 0.0}
        
        try:
            # Try to extract JSON from markdown code blocks
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            
            # Add metadata about the response
            parsed["_meta"] = {
                "model": response.model,
                "retry_count": response.retry_count,
                "latency_ms": response.latency_ms,
                "fallback_used": response.fallback_used
            }
            
            if response.uncertainty:
                parsed["_uncertainty"] = {
                    "confidence_score": response.uncertainty.confidence_score,
                    "confidence_level": response.uncertainty.confidence_level.value,
                    "reasoning_steps": response.uncertainty.reasoning_steps,
                    "evidence_strength": response.uncertainty.evidence_strength
                }
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {
                "error": "Failed to parse response",
                "raw_content": response.content[:500],  # First 500 chars
                "confidence": 0.0
            }
