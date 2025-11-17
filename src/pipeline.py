"""
Self-Critique Chain Pipeline implementation.

This module contains the main pipeline class that orchestrates the three-stage
execution flow for research paper summarization with automatic critique and revision.
The implementation follows production-grade principles including comprehensive error
handling, metrics tracking, and integration with MLflow for experiment tracking.
"""

import time
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import anthropic
from src.prompts import get_prompt_template
from src.utils import extract_xml_content, validate_api_key, format_timestamp


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass


class APIError(PipelineError):
    """Exception raised when Claude API calls fail."""
    pass


class ValidationError(PipelineError):
    """Exception raised when input validation fails."""
    pass


class SelfCritiquePipeline:
    """
    Self-Critique Chain Pipeline for research paper summarization.
    
    This class implements a three-stage pipeline that generates initial summaries,
    performs self-critique to identify issues, and produces revised versions based
    on the critique feedback. The pipeline uses Claude AI with optimized temperature
    settings for each stage and structured XML output parsing.
    
    The implementation follows the Single Responsibility Principle with each stage
    handling a distinct aspect of the summarization process. The Open-Closed Principle
    is achieved through configuration-based prompt templates that can be extended
    without modifying the core logic.
    
    Attributes:
        client: Anthropic API client instance
        model: Claude model identifier for API calls
        max_tokens: Maximum tokens per API response
        system_prompt: System-level instructions for Claude
        metrics: List of metrics collected during pipeline execution
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        """
        Initialize the Self-Critique Pipeline.
        
        Args:
            api_key: Anthropic API key for authentication
            model: Claude model version to use for all stages
            max_tokens: Maximum tokens allowed per response
            
        Raises:
            ValidationError: If API key format is invalid
        """
        if not validate_api_key(api_key):
            raise ValidationError(
                "Invalid API key format. Anthropic API keys should start with 'sk-ant-'"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = get_prompt_template("system")
        self.metrics = []
    
    def _call_claude(
        self,
        prompt: str,
        temperature: float = 0.3,
        stage_name: str = "unknown"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute a single Claude API call with comprehensive error handling.
        
        This method wraps the Anthropic API call with timing measurements, error
        handling, and metrics collection. The temperature parameter is adjusted
        per stage to optimize for either factual accuracy or creative critique.
        
        Args:
            prompt: User prompt to send to Claude
            temperature: Sampling temperature between zero and one
            stage_name: Identifier for the pipeline stage making this call
            
        Returns:
            Tuple containing the response text and a metrics dictionary with
            timestamp, token counts, latency, and other performance indicators
            
        Raises:
            APIError: If the API call fails for any reason
        """
        start_time = time.time()
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
        except anthropic.APIError as e:
            raise APIError(f"Claude API call failed: {str(e)}") from e
        except Exception as e:
            raise APIError(f"Unexpected error during API call: {str(e)}") from e
        
        latency = time.time() - start_time
        
        metrics = {
            "timestamp": format_timestamp(),
            "stage": stage_name,
            "model": self.model,
            "temperature": temperature,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
            "latency_seconds": round(latency, 3),
            "stop_reason": message.stop_reason,
        }
        
        self.metrics.append(metrics)
        
        return message.content[0].text, metrics
    
    def _execute_stage1_summarization(self, paper_text: str) -> Dict[str, Any]:
        """
        Execute Stage 1 which generates the initial summary.
        
        This stage uses a temperature of 0.3 to ensure factual accuracy and minimize
        hallucinations. The output is parsed to extract the summary content from XML
        tags while preserving the thinking process for debugging purposes.
        
        Args:
            paper_text: Full text of the research paper to summarize
            
        Returns:
            Dictionary containing the summary, thinking process, response, and metrics
        """
        prompt_template = get_prompt_template("stage1")
        prompt = prompt_template.format(paper_text=paper_text)
        
        response, metrics = self._call_claude(
            prompt=prompt,
            temperature=0.3,
            stage_name="stage1_summarization"
        )
        
        summary = extract_xml_content(response, "output")
        thinking = extract_xml_content(response, "thinking")
        
        if not summary:
            raise PipelineError(
                "Stage 1 failed to generate summary with proper XML tags. "
                "Check the response format."
            )
        
        return {
            "summary": summary,
            "thinking": thinking,
            "full_response": response,
            "metrics": metrics
        }
    
    def _execute_stage2_critique(
        self,
        paper_text: str,
        summary: str
    ) -> Dict[str, Any]:
        """
        Execute Stage 2 which critiques the initial summary.
        
        This stage uses a slightly higher temperature of 0.5 to encourage more
        creative and thorough critical analysis. The critique identifies issues
        across four dimensions with evidence-based feedback for each problem.
        
        Args:
            paper_text: Original research paper text
            summary: Summary generated in Stage 1
            
        Returns:
            Dictionary containing the critique, thinking process, response, and metrics
        """
        prompt_template = get_prompt_template("stage2")
        prompt = prompt_template.format(
            paper_text=paper_text,
            summary=summary
        )
        
        response, metrics = self._call_claude(
            prompt=prompt,
            temperature=0.5,
            stage_name="stage2_critique"
        )
        
        critique = extract_xml_content(response, "critique")
        thinking = extract_xml_content(response, "thinking")
        
        if not critique:
            raise PipelineError(
                "Stage 2 failed to generate critique with proper XML tags. "
                "Check the response format."
            )
        
        return {
            "critique": critique,
            "thinking": thinking,
            "full_response": response,
            "metrics": metrics
        }
    
    def _execute_stage3_revision(
        self,
        paper_text: str,
        summary: str,
        critique: str
    ) -> Dict[str, Any]:
        """
        Execute Stage 3 which revises the summary based on critique.
        
        This stage returns to a temperature of 0.3 to ensure the revised summary
        maintains factual accuracy while addressing all identified issues. The
        revision includes a reflection section explaining what changed and why.
        
        Args:
            paper_text: Original research paper text
            summary: Summary from Stage 1
            critique: Critique from Stage 2
            
        Returns:
            Dictionary containing revised summary, reflection, response, and metrics
        """
        prompt_template = get_prompt_template("stage3")
        prompt = prompt_template.format(
            paper_text=paper_text,
            summary=summary,
            critique=critique
        )
        
        response, metrics = self._call_claude(
            prompt=prompt,
            temperature=0.3,
            stage_name="stage3_revision"
        )
        
        revised_summary = extract_xml_content(response, "output")
        reflection = extract_xml_content(response, "reflection")
        thinking = extract_xml_content(response, "thinking")
        
        if not revised_summary:
            raise PipelineError(
                "Stage 3 failed to generate revised summary with proper XML tags. "
                "Check the response format."
            )
        
        return {
            "revised_summary": revised_summary,
            "reflection": reflection,
            "thinking": thinking,
            "full_response": response,
            "metrics": metrics
        }
    
    def run_pipeline(
        self,
        paper_text: str,
        mlflow_tracking: bool = False,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete three-stage Self-Critique Chain pipeline.
        
        This method orchestrates the entire workflow from initial summarization through
        critique to final revision. It collects comprehensive metrics at each stage and
        optionally logs everything to MLflow for experiment tracking and reproducibility.
        
        The pipeline implements a fail-fast strategy where any stage failure immediately
        raises an exception rather than attempting to continue with partial results. This
        ensures that downstream stages always receive valid inputs and prevents cascading
        errors from producing misleading outputs.
        
        Args:
            paper_text: Full text of the research paper to summarize
            mlflow_tracking: Enable MLflow experiment tracking if True
            experiment_name: Optional experiment name for MLflow runs
            
        Returns:
            Dictionary containing all outputs and metrics from each stage including
            the original summary, critique analysis, revised summary, reflection on
            changes made, and comprehensive performance metrics
            
        Raises:
            ValidationError: If input paper text is empty or invalid
            APIError: If any Claude API call fails
            PipelineError: If any stage fails to produce valid output
        """
        if not paper_text or not isinstance(paper_text, str):
            raise ValidationError("Paper text must be a non-empty string")
        
        if len(paper_text.strip()) < 100:
            raise ValidationError(
                "Paper text is too short. Provide at least 100 characters of content."
            )
        
        results = {
            "pipeline_start": format_timestamp(),
            "model": self.model,
            "paper_length": len(paper_text)
        }
        
        mlflow_run = None
        if mlflow_tracking:
            try:
                import mlflow
                if experiment_name:
                    mlflow.set_experiment(experiment_name)
                mlflow_run = mlflow.start_run()
                mlflow.log_param("model", self.model)
                mlflow.log_param("max_tokens", self.max_tokens)
                mlflow.log_param("paper_length", len(paper_text))
            except ImportError:
                print("Warning: MLflow not installed. Tracking disabled.")
                mlflow_tracking = False
        
        try:
            print("\n" + "="*60)
            print("SELF-CRITIQUE CHAIN PIPELINE EXECUTION")
            print("="*60)
            
            print("\n🔄 Stage 1: Generating initial summary...")
            stage1_results = self._execute_stage1_summarization(paper_text)
            results.update({
                "summary": stage1_results["summary"],
                "stage1_thinking": stage1_results["thinking"],
                "stage1_response": stage1_results["full_response"],
                "stage1_metrics": stage1_results["metrics"]
            })
            print(f"✅ Stage 1 complete ({stage1_results['metrics']['output_tokens']} tokens, "
                  f"{stage1_results['metrics']['latency_seconds']:.2f}s)")
            
            print("\n🔍 Stage 2: Critiquing summary...")
            stage2_results = self._execute_stage2_critique(
                paper_text=paper_text,
                summary=stage1_results["summary"]
            )
            results.update({
                "critique": stage2_results["critique"],
                "stage2_thinking": stage2_results["thinking"],
                "stage2_response": stage2_results["full_response"],
                "stage2_metrics": stage2_results["metrics"]
            })
            print(f"✅ Stage 2 complete ({stage2_results['metrics']['output_tokens']} tokens, "
                  f"{stage2_results['metrics']['latency_seconds']:.2f}s)")
            
            print("\n✏️ Stage 3: Revising summary...")
            stage3_results = self._execute_stage3_revision(
                paper_text=paper_text,
                summary=stage1_results["summary"],
                critique=stage2_results["critique"]
            )
            results.update({
                "revised_summary": stage3_results["revised_summary"],
                "reflection": stage3_results["reflection"],
                "stage3_thinking": stage3_results["thinking"],
                "stage3_response": stage3_results["full_response"],
                "stage3_metrics": stage3_results["metrics"]
            })
            print(f"✅ Stage 3 complete ({stage3_results['metrics']['output_tokens']} tokens, "
                  f"{stage3_results['metrics']['latency_seconds']:.2f}s)")
            
            total_metrics = self._calculate_total_metrics()
            results["total_metrics"] = total_metrics
            results["pipeline_end"] = format_timestamp()
            
            print("\n" + "="*60)
            print("PIPELINE EXECUTION COMPLETE")
            print("="*60)
            print(f"Total tokens: {total_metrics['total_tokens']}")
            print(f"Total latency: {total_metrics['total_latency_seconds']:.2f}s")
            print(f"Stages completed: {total_metrics['stage_count']}")
            print("="*60 + "\n")
            
            if mlflow_tracking and mlflow_run:
                self._log_to_mlflow(results)
            
        except Exception as e:
            if mlflow_tracking and mlflow_run:
                import mlflow
                mlflow.log_param("error", str(e))
                mlflow.end_run(status="FAILED")
            raise
        finally:
            if mlflow_tracking and mlflow_run:
                import mlflow
                mlflow.end_run()
        
        return results
    
    def _calculate_total_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all pipeline stages."""
        total_input_tokens = sum(m["input_tokens"] for m in self.metrics)
        total_output_tokens = sum(m["output_tokens"] for m in self.metrics)
        total_latency = sum(m["latency_seconds"] for m in self.metrics)
        
        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_latency_seconds": round(total_latency, 3),
            "stage_count": len(self.metrics),
            "average_latency_per_stage": round(total_latency / len(self.metrics), 3) if self.metrics else 0
        }
    
    def _log_to_mlflow(self, results: Dict[str, Any]) -> None:
        """Log pipeline results and artifacts to MLflow."""
        try:
            import mlflow
            
            mlflow.log_metrics(results["total_metrics"])
            
            for stage_num in range(1, 4):
                stage_metrics = results.get(f"stage{stage_num}_metrics", {})
                if stage_metrics:
                    mlflow.log_metrics({
                        f"stage{stage_num}_tokens": stage_metrics.get("total_tokens", 0),
                        f"stage{stage_num}_latency": stage_metrics.get("latency_seconds", 0)
                    })
            
            with open("temp_summary.txt", "w", encoding="utf-8") as f:
                f.write(results.get("summary", ""))
            mlflow.log_artifact("temp_summary.txt", "summaries")
            
            with open("temp_revised.txt", "w", encoding="utf-8") as f:
                f.write(results.get("revised_summary", ""))
            mlflow.log_artifact("temp_revised.txt", "summaries")
            
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {str(e)}")