"""Configuration management for the evaluation framework."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class APIConfig:
    """Configuration for LLM API access."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2000
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            default_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("EVALUATION_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000"))
        )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    enable_deterministic: bool = True
    enable_hallucination_detection: bool = False
    enable_completeness_check: bool = False
    enable_clinical_accuracy: bool = False
    sample_size: Optional[int] = None
    output_dir: str = "results"
    cache_dir: str = "data"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Load configuration from environment variables."""
        return cls(
            enable_deterministic=os.getenv("ENABLE_DETERMINISTIC", "true").lower() == "true",
            enable_hallucination_detection=os.getenv("ENABLE_LLM_JUDGE", "false").lower() == "true",
            enable_completeness_check=os.getenv("ENABLE_LLM_JUDGE", "false").lower() == "true",
            enable_clinical_accuracy=os.getenv("ENABLE_LLM_JUDGE", "false").lower() == "true",
            sample_size=int(os.getenv("SAMPLE_SIZE")) if os.getenv("SAMPLE_SIZE") else None,
            output_dir=os.getenv("OUTPUT_DIR", "results"),
            cache_dir=os.getenv("CACHE_DIR", "data"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )


@dataclass
class Config:
    """Main configuration object."""
    api: APIConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment and optional .env file."""
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        
        return cls(
            api=APIConfig.from_env(),
            evaluation=EvaluationConfig.from_env()
        )


# Global configuration instance
config = Config.load(".env")

