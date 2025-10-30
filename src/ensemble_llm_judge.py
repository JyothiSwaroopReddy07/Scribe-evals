"""Ensemble LLM Judge with multiple models, voting, and robust error handling."""

import asyncio
import json
import logging
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

from .confidence_scorer import ConfidenceScorer, ConfidenceScore


logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Voting strategies for ensemble."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    UNANIMOUS = "unanimous"


@dataclass
class LLMResponse:
    """Enhanced response from LLM judge."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    confidence: Optional[float] = None
    latency: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleResult:
    """Result from ensemble evaluation."""
    consensus_result: Any
    individual_results: List[LLMResponse]
    confidence_score: ConfidenceScore
    voting_details: Dict[str, Any]
    metadata: Dict[str, Any]


class RetryStrategy:
    """Retry strategy with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay


class EnhancedLLMJudge:
    """Enhanced LLM Judge with retry, fallback, and error handling."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        fallback_models: Optional[List[str]] = None,
        timeout: float = 30.0
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.fallback_models = fallback_models or []
        
        # Initialize API client
        self._init_client(model, api_key)
        
        # Metrics
        self.call_count = 0
        self.error_count = 0
        self.total_latency = 0.0
    
    def _init_client(self, model: str, api_key: Optional[str] = None):
        """Initialize API client for given model."""
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
        response_format: Optional[str] = None
    ) -> LLMResponse:
        """
        Call LLM with retry and fallback logic.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            response_format: Optional format (e.g., "json")
            
        Returns:
            LLMResponse object
        """
        models_to_try = [self.model] + self.fallback_models
        
        for model_idx, model in enumerate(models_to_try):
            for attempt in range(self.retry_strategy.max_retries):
                try:
                    logger.info(
                        f"Calling {model} (attempt {attempt + 1}/{self.retry_strategy.max_retries})"
                    )
                    
                    # Update client if needed
                    if model != self.model and model_idx > 0:
                        self._init_client(model)
                    
                    start_time = time.time()
                    
                    if self.provider == "openai":
                        response = self._call_openai(
                            system_prompt, user_prompt, response_format
                        )
                    elif self.provider == "anthropic":
                        response = self._call_anthropic(
                            system_prompt, user_prompt
                        )
                    
                    latency = time.time() - start_time
                    response.latency = latency
                    
                    self.call_count += 1
                    self.total_latency += latency
                    
                    logger.info(f"Successfully called {model} in {latency:.2f}s")
                    return response
                    
                except Exception as e:
                    self.error_count += 1
                    logger.warning(
                        f"Error calling {model} (attempt {attempt + 1}): {e}"
                    )
                    
                    # Check if we should retry
                    if attempt < self.retry_strategy.max_retries - 1:
                        delay = self.retry_strategy.get_delay(attempt)
                        logger.info(f"Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    elif model_idx < len(models_to_try) - 1:
                        logger.info(f"Falling back to next model: {models_to_try[model_idx + 1]}")
                        break
                    else:
                        logger.error(f"All retries and fallbacks exhausted")
                        # Return empty response
                        return LLMResponse(
                            content="",
                            model=model,
                            usage=None,
                            metadata={"error": str(e)}
                        )
        
        # Should not reach here
        return LLMResponse(
            content="",
            model=self.model,
            usage=None,
            metadata={"error": "All attempts failed"}
        )
    
    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None
    ) -> LLMResponse:
        """Call OpenAI API with timeout."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
        
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
    
    def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> LLMResponse:
        """Call Anthropic API with timeout."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=self.timeout
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        )
    
    def parse_json_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse JSON response with error handling."""
        try:
            content = response.content.strip()
            
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.debug(f"Content: {response.content[:500]}")
            
            # Try to salvage partial JSON
            try:
                # Find JSON-like content
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                pass
            
            return {}
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get judge performance metrics."""
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.call_count, 1),
            "average_latency": self.total_latency / max(self.call_count, 1),
            "total_latency": self.total_latency
        }


class EnsembleLLMJudge:
    """Ensemble of multiple LLM judges with voting."""
    
    def __init__(
        self,
        models: List[str],
        voting_strategy: VotingStrategy = VotingStrategy.CONFIDENCE_WEIGHTED,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        enable_parallel: bool = False
    ):
        self.models = models
        self.voting_strategy = voting_strategy
        self.enable_parallel = enable_parallel
        
        # Initialize judges
        self.judges = [
            EnhancedLLMJudge(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                retry_strategy=RetryStrategy(max_retries=2)
            )
            for model in models
        ]
        
        self.confidence_scorer = ConfidenceScorer()
    
    def evaluate_ensemble(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None,
        parse_fn: Optional[Callable] = None
    ) -> EnsembleResult:
        """
        Evaluate with ensemble of models.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            response_format: Optional format
            parse_fn: Function to parse and extract answer
            
        Returns:
            EnsembleResult with consensus
        """
        # Get responses from all judges
        responses = []
        for judge in self.judges:
            response = judge.evaluate(system_prompt, user_prompt, response_format)
            responses.append(response)
        
        # Parse responses
        if parse_fn:
            parsed_results = [parse_fn(r) for r in responses]
        elif response_format == "json":
            parsed_results = [
                self.judges[0].parse_json_response(r) for r in responses
            ]
        else:
            parsed_results = [r.content for r in responses]
        
        # Apply voting strategy
        consensus, voting_details = self._apply_voting(
            parsed_results, responses
        )
        
        # Compute ensemble confidence
        scores = [r.confidence for r in responses if r.confidence is not None]
        if not scores:
            scores = None
        
        confidence_score = self.confidence_scorer.compute_ensemble_confidence(
            predictions=parsed_results,
            scores=scores
        )
        
        # Metadata
        metadata = {
            "n_models": len(self.models),
            "models": self.models,
            "voting_strategy": self.voting_strategy.value,
            "total_tokens": sum(
                r.usage.get("total_tokens", 0) 
                for r in responses if r.usage
            ),
            "average_latency": sum(
                r.latency for r in responses if r.latency
            ) / len(responses)
        }
        
        return EnsembleResult(
            consensus_result=consensus,
            individual_results=responses,
            confidence_score=confidence_score,
            voting_details=voting_details,
            metadata=metadata
        )
    
    def _apply_voting(
        self,
        parsed_results: List[Any],
        responses: List[LLMResponse]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply voting strategy to get consensus."""
        if self.voting_strategy == VotingStrategy.MAJORITY_VOTE:
            return self._majority_vote(parsed_results)
        
        elif self.voting_strategy == VotingStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_vote(parsed_results, responses)
        
        elif self.voting_strategy == VotingStrategy.UNANIMOUS:
            return self._unanimous_vote(parsed_results)
        
        else:
            # Default to majority vote
            return self._majority_vote(parsed_results)
    
    def _majority_vote(
        self,
        parsed_results: List[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Simple majority voting."""
        # Convert to strings for hashing
        result_strs = [json.dumps(r, sort_keys=True) for r in parsed_results]
        counter = Counter(result_strs)
        most_common = counter.most_common(1)[0]
        
        consensus_str = most_common[0]
        count = most_common[1]
        
        consensus = json.loads(consensus_str)
        
        details = {
            "vote_counts": dict(counter),
            "winner_count": count,
            "total_votes": len(parsed_results),
            "agreement_rate": count / len(parsed_results)
        }
        
        return consensus, details
    
    def _confidence_weighted_vote(
        self,
        parsed_results: List[Any],
        responses: List[LLMResponse]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Confidence-weighted voting."""
        # Extract confidences
        confidences = []
        for r in responses:
            if r.confidence is not None:
                confidences.append(r.confidence)
            else:
                # Default confidence
                confidences.append(0.7)
        
        # Weight votes by confidence
        result_strs = [json.dumps(r, sort_keys=True) for r in parsed_results]
        weighted_votes = {}
        
        for result_str, confidence in zip(result_strs, confidences):
            if result_str not in weighted_votes:
                weighted_votes[result_str] = 0.0
            weighted_votes[result_str] += confidence
        
        # Get winner
        winner = max(weighted_votes.items(), key=lambda x: x[1])
        consensus = json.loads(winner[0])
        
        details = {
            "weighted_votes": weighted_votes,
            "winner_weight": winner[1],
            "total_weight": sum(weighted_votes.values())
        }
        
        return consensus, details
    
    def _unanimous_vote(
        self,
        parsed_results: List[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Unanimous voting (all must agree)."""
        result_strs = [json.dumps(r, sort_keys=True) for r in parsed_results]
        
        if len(set(result_strs)) == 1:
            # Unanimous
            consensus = parsed_results[0]
            is_unanimous = True
        else:
            # No consensus, return majority
            consensus, _ = self._majority_vote(parsed_results)
            is_unanimous = False
        
        details = {
            "is_unanimous": is_unanimous,
            "unique_results": len(set(result_strs))
        }
        
        return consensus, details
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get metrics from all judges."""
        all_metrics = {}
        
        for i, judge in enumerate(self.judges):
            metrics = judge.get_metrics()
            all_metrics[f"model_{i}_{judge.model}"] = metrics
        
        # Aggregate
        total_calls = sum(m["call_count"] for m in all_metrics.values())
        total_errors = sum(m["error_count"] for m in all_metrics.values())
        avg_latency = sum(
            m["average_latency"] * m["call_count"] 
            for m in all_metrics.values()
        ) / max(total_calls, 1)
        
        all_metrics["aggregate"] = {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / max(total_calls, 1),
            "average_latency": avg_latency
        }
        
        return all_metrics

