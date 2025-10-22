# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-22

### Added
- Initial release of SOAP Note Evaluation Framework
- Deterministic evaluation metrics (ROUGE, BERTScore, structure checks)
- LLM-based evaluators (hallucination detection, completeness, clinical accuracy)
- Support for multiple data sources (Omi-Health, adesouza1, custom JSON)
- Interactive Streamlit dashboard for results visualization
- Docker support with multi-container orchestration
- Comprehensive logging and configuration management
- CI/CD integration with regression detection
- Evaluated on 9,808 real clinical SOAP notes
- Performance: 25+ notes/second for deterministic metrics
- Cost-effective LLM sampling strategy for production monitoring

### Features
- Hybrid two-tier evaluation approach
- Non-reference-based evaluation for production use
- Reference-based evaluation for benchmarking
- Multi-dimensional quality scoring
- Extensible evaluator architecture
- Batch processing support
- JSON and CSV output formats
- API support for OpenAI and Anthropic

### Documentation
- Comprehensive README with usage examples
- Technical documentation
- Docker deployment guide
- API reference
- Contributing guidelines

### Testing
- Unit tests for all evaluators
- Integration tests for pipeline
- Validation on synthetic test cases with known issues
- Benchmarked on 9,808 real clinical notes

## [Unreleased]

### Planned
- Fine-tuned evaluation models for faster inference
- Clinical NER integration (BioBERT, ClinicalBERT)
- Multi-model ensembling for higher reliability
- Real-time streaming support
- Specialty-specific evaluators
- Human feedback loop integration
- HIPAA-compliant deployment options
- Additional LLM provider support

