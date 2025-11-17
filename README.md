# Self-Critique Chain Pipeline

Production-ready implementation of the Self-Critique Chain pattern for automated research paper summarization using Claude AI. This system implements a three-stage pipeline that generates, critiques, and revises summaries with automatic quality assessment and anomaly detection.

## Overview

The Self-Critique Chain Pipeline addresses the challenge of producing high-quality, accurate summaries of research papers through an iterative refinement process. Traditional single-shot summarization approaches often produce outputs with inconsistencies, missing details, or misrepresented findings. This pipeline solves these problems by implementing a Chain-of-Verification pattern where Claude AI systematically evaluates and improves its own outputs.

## Key Features

**Three-Stage Processing Pipeline** implements distinct stages for summarization, critique, and revision with proper separation of concerns. Each stage operates with optimized temperature settings and structured XML output formatting to ensure consistency and reliability.

**Automatic Quality Assessment** evaluates outputs across four dimensions including accuracy, completeness, clarity, and coherence. The system generates quantitative scores on a zero to ten scale and provides detailed evidence-based feedback for each identified issue.

**Production-Grade Monitoring** tracks all API requests with comprehensive metrics including token usage, latency, and user feedback. The system detects anomalies such as latency spikes or declining satisfaction scores and triggers alerts when thresholds are exceeded.

**RESTful API Interface** exposes the pipeline through FastAPI endpoints with automatic request validation, comprehensive error handling, and OpenAPI documentation generation for easy integration with existing systems.

**MLflow Integration** enables experiment tracking with automatic logging of parameters, metrics, and artifacts. This supports reproducibility and facilitates comparison between different prompt versions or model configurations.

## Architecture

The system follows clean architecture principles with clear separation between core business logic, API layer, and infrastructure concerns. The pipeline module handles the three-stage execution flow, the monitoring module provides observability, and the API module exposes functionality through HTTP endpoints.

The design adheres to SOLID principles through dependency injection, interface segregation, and the open-closed principle. Components can be extended through configuration without modifying core logic, and all dependencies point inward toward abstractions rather than concrete implementations.

## Installation

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/arec1b0/self-critique-pipeline.git
cd self-critique-pipeline
```

Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

Install the package and dependencies:
```bash
pip install -e .
```

For development with testing and code quality tools:
```bash
pip install -e ".[dev]"
```

## Configuration

Copy the example environment file and configure your API credentials:
```bash
copy .env.example .env
```

Edit the `.env` file and add your Anthropic API key:
```env
ANTHROPIC_API_KEY=your_actual_api_key_here
```

The configuration file at `config/config.yaml` contains additional settings for prompt templates, model parameters, and monitoring thresholds. Modify these values to customize the pipeline behavior for your specific requirements.

## Usage

### Basic Example

The following example demonstrates the core pipeline functionality with a sample research paper:
```python
from src.pipeline import SelfCritiquePipeline

pipeline = SelfCritiquePipeline(
    api_key="your_api_key",
    model="claude-sonnet-4-20250514"
)

paper_text = """Your research paper content here..."""

results = pipeline.run_pipeline(
    paper_text=paper_text,
    mlflow_tracking=True
)

print(results["revised_summary"])
```

### Running the API Server

Start the FastAPI server for HTTP access to the pipeline:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation will be available at `http://localhost:8000/docs` with interactive endpoint testing through the Swagger UI interface.

### Monitoring and Analytics

The monitoring module tracks all requests and provides anomaly detection:
```python
from src.monitoring import PromptMonitor

monitor = PromptMonitor(baseline_latency=2.0)

# Log requests during pipeline execution
monitor.log_request(prompt, response, metrics, user_feedback=4)

# Detect anomalies in performance
anomalies = monitor.detect_anomalies()

# Export logs for analysis
df = monitor.export_to_dataframe()
```

## Testing

Execute the test suite with coverage reporting:
```bash
pytest tests/ --cov=src --cov-report=html
```

The coverage report will be generated in the `htmlcov` directory. Open `htmlcov/index.html` in a browser to view detailed coverage metrics.

## Project Structure

The repository organization follows a modular structure that separates concerns and facilitates maintenance:
```
self-critique-pipeline/
├── src/              # Core pipeline implementation
├── api/              # FastAPI REST endpoints
├── config/           # Configuration files and prompt templates
├── tests/            # Test suite with fixtures
├── examples/         # Usage examples and demonstrations
└── notebooks/        # Jupyter notebooks for experimentation
```

## Performance Characteristics

The pipeline demonstrates significant improvements over single-shot prompting approaches. Internal testing shows a reduction in hallucinations ranging from fifteen to twenty-five percent through the critique and revision stages. Average end-to-end latency for a typical research paper abstract is approximately six to eight seconds across all three stages.

Token consumption varies based on paper length but typically ranges from three thousand to five thousand total tokens including both input and output across all stages. The system scales horizontally through the FastAPI interface and can handle concurrent requests limited only by your Anthropic API rate limits.

## Contributing

Contributions are welcome and encouraged. Please review the `CONTRIBUTING.md` file for guidelines on code style, testing requirements, and the pull request process. All contributions should include appropriate tests and documentation updates.

## License

This project is licensed under the MIT License. See the `LICENSE` file for complete terms and conditions.

## Acknowledgments

This implementation is based on research into Chain-of-Verification patterns and self-critique mechanisms for large language models. The architecture follows production-grade MLOps practices including reproducibility, observability, and reversibility principles.

## Support

For bug reports and feature requests, please use the GitHub Issues tracker. For questions and discussions, consult the documentation or reach out through the repository discussions section.