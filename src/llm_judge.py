"""LLM Judge interface for SOAP note evaluation."""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class LLMResponse:
    """Response from LLM judge."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class LLMJudge:
    """Interface for LLM-based evaluation."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
        response_format: Optional[str] = None
    ) -> LLMResponse:
        """
        Call LLM for evaluation.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query/content to evaluate
            response_format: Optional format specification (e.g., "json")
            
        Returns:
            LLMResponse object
        """
        try:
            if self.provider == "openai":
                return self._call_openai(system_prompt, user_prompt, response_format)
            elif self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            # Return empty response on error
            return LLMResponse(
                content="",
                model=self.model,
                usage=None
            )
    
    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None
    ) -> LLMResponse:
        """Call OpenAI API."""
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
        """Parse JSON response from LLM."""
        try:
            # Try to extract JSON from markdown code blocks
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}", exc_info=True)
            return {}


class PromptTemplates:
    """Prompt templates for LLM-based evaluation."""
    
    @staticmethod
    def hallucination_detection() -> tuple[str, str]:
        """Prompt for detecting hallucinated facts."""
        system_prompt = """You are a medical documentation expert tasked with identifying hallucinated or unsupported facts in clinical SOAP notes.

            Your job is to compare a generated SOAP note against the source transcript and identify any information in the note that is NOT supported by the transcript.

            A fact is considered hallucinated if:
            1. It's explicitly stated in the note but not mentioned in the transcript
            2. It contradicts information in the transcript
            3. It makes specific claims (dosages, dates, measurements) not present in the source

            Do NOT flag:
            - Standard medical formatting or SOAP structure
            - Reasonable clinical terminology for described symptoms
            - Normal ranges or standard procedures implied by the context

            Respond in JSON format with:
            {
            "hallucinations": [
                {
                "fact": "specific hallucinated statement",
                "severity": "high/medium/low",
                "explanation": "why this is unsupported",
                "location": "which section (S/O/A/P)"
                }
            ],
            "hallucination_score": 0.0-1.0 (0 = many hallucinations, 1 = none),
            "confidence": 0.0-1.0
            }"""
                    
        user_template = """Transcript:
            {transcript}

            Generated SOAP Note:
            {generated_note}

            Identify any hallucinated or unsupported facts in the note."""
        
        return system_prompt, user_template
    
    @staticmethod
    def completeness_check() -> tuple[str, str]:
        """Prompt for checking completeness."""
        system_prompt = """You are a medical documentation expert tasked with identifying critical information from patient encounters that may be missing from SOAP notes.

            Your job is to review the transcript and identify any clinically significant information that should have been documented but is missing from the generated note.

            Focus on:
            1. Key symptoms or complaints mentioned by the patient
            2. Important physical examination findings
            3. Relevant medical history
            4. Diagnoses or clinical impressions
            5. Treatment plans or recommendations
            6. Follow-up instructions

            Do NOT flag:
            - Minor conversational elements
            - Redundant information
            - Information adequately captured in different wording

            Respond in JSON format with:
            {
            "missing_items": [
                {
                "information": "what's missing",
                "severity": "critical/high/medium/low",
                "explanation": "why this is important",
                "location": "where it should appear (S/O/A/P)"
                }
            ],
            "completeness_score": 0.0-1.0 (0 = major gaps, 1 = complete),
            "confidence": 0.0-1.0
            }"""
                    
        user_template = """Transcript:
            {transcript}

            Generated SOAP Note:
            {generated_note}

            Identify any critical information from the transcript that is missing from the note."""
        
        return system_prompt, user_template
    
    @staticmethod
    def clinical_accuracy() -> tuple[str, str]:
        """Prompt for checking clinical accuracy."""
        system_prompt = """You are a clinical documentation expert tasked with identifying medical inaccuracies or misleading statements in SOAP notes.

            Your job is to review the generated note for:
            1. Medically incorrect statements
            2. Inappropriate clinical conclusions
            3. Misrepresentation of symptoms or findings
            4. Incorrect medical terminology usage
            5. Dangerous or inappropriate treatment recommendations

            Consider:
            - Clinical appropriateness of assessments
            - Logical consistency between sections
            - Proper use of medical terminology
            - Safety of recommendations

            Respond in JSON format with:
            {
            "accuracy_issues": [
                {
                "issue": "description of the problem",
                "severity": "critical/high/medium/low",
                "explanation": "why this is problematic",
                "location": "where in the note",
                "correction": "what should be stated instead (if applicable)"
                }
            ],
            "accuracy_score": 0.0-1.0 (0 = major issues, 1 = accurate),
            "confidence": 0.0-1.0
            }"""
                    
        user_template = """Transcript:
            {transcript}

            Generated SOAP Note:
            {generated_note}

            Identify any clinical accuracy issues or medically problematic statements in the note."""
        
        return system_prompt, user_template

