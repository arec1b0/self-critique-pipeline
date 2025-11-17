"""
Pydantic schemas for API request and response validation.

This module defines all data models used by the FastAPI endpoints to ensure
type safety, automatic validation, and comprehensive API documentation. The
schemas follow the Single Responsibility Principle with each model handling
a specific aspect of the API contract.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class PipelineRequest(BaseModel):
    """
    Request schema for pipeline execution endpoint.
    
    This model validates incoming requests to ensure they contain all required
    fields with appropriate constraints. The validation helps prevent common
    errors such as empty paper text or invalid configuration parameters.
    """
    
    paper_text: str = Field(
        ...,
        min_length=100,
        max_length=50000,
        description="Full text of the research paper to summarize. Must contain at least 100 characters of meaningful content."
    )
    
    enable_mlflow: bool = Field(
        default=False,
        description="Enable MLflow experiment tracking for this pipeline execution"
    )
    
    experiment_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional experiment name for MLflow tracking. If not provided, uses the default experiment."
    )
    
    model: Optional[str] = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model version to use for pipeline execution"
    )
    
    @validator("paper_text")
    def validate_paper_text(cls, v):
        """Validate that paper text contains meaningful content."""
        if not v or not v.strip():
            raise ValueError("Paper text cannot be empty or contain only whitespace")
        
        if len(v.strip()) < 100:
            raise ValueError("Paper text must contain at least 100 characters of meaningful content")
        
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "paper_text": "Title: Attention Is All You Need\n\nAbstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "enable_mlflow": False,
                "experiment_name": "research-paper-summarization",
                "model": "claude-sonnet-4-20250514"
            }
        }


class MetricsResponse(BaseModel):
    """Schema for performance metrics from a single pipeline stage."""
    
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp of the request")
    stage: str = Field(..., description="Pipeline stage identifier")
    model: str = Field(..., description="Claude model version used")
    temperature: float = Field(..., description="Sampling temperature used for generation")
    input_tokens: int = Field(..., description="Number of input tokens consumed")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    total_tokens: int = Field(..., description="Total tokens consumed for this stage")
    latency_seconds: float = Field(..., description="Request latency in seconds")
    stop_reason: str = Field(..., description="Reason why the generation stopped")


class TotalMetricsResponse(BaseModel):
    """Schema for aggregate metrics across all pipeline stages."""
    
    total_input_tokens: int = Field(..., description="Total input tokens across all stages")
    total_output_tokens: int = Field(..., description="Total output tokens across all stages")
    total_tokens: int = Field(..., description="Total tokens consumed by entire pipeline")
    total_latency_seconds: float = Field(..., description="Total latency across all stages")
    stage_count: int = Field(..., description="Number of stages executed")
    average_latency_per_stage: float = Field(..., description="Average latency per stage")


class PipelineResponse(BaseModel):
    """
    Response schema for successful pipeline execution.
    
    This model structures the complete output from the three-stage pipeline including
    all generated content, intermediate products, and comprehensive performance metrics.
    The structured format supports both human consumption and programmatic processing.
    """
    
    pipeline_start: str = Field(..., description="Pipeline execution start timestamp")
    pipeline_end: str = Field(..., description="Pipeline execution completion timestamp")
    model: str = Field(..., description="Claude model version used")
    paper_length: int = Field(..., description="Length of input paper in characters")
    
    summary: str = Field(..., description="Initial summary generated in Stage 1")
    critique: str = Field(..., description="Self-critique analysis from Stage 2")
    revised_summary: str = Field(..., description="Final revised summary from Stage 3")
    reflection: str = Field(..., description="Reflection on changes made during revision")
    
    stage1_metrics: MetricsResponse = Field(..., description="Performance metrics for Stage 1")
    stage2_metrics: MetricsResponse = Field(..., description="Performance metrics for Stage 2")
    stage3_metrics: MetricsResponse = Field(..., description="Performance metrics for Stage 3")
    total_metrics: TotalMetricsResponse = Field(..., description="Aggregate metrics for entire pipeline")
    
    class Config:
        schema_extra = {
            "example": {
                "pipeline_start": "2024-11-17T10:30:00Z",
                "pipeline_end": "2024-11-17T10:30:15Z",
                "model": "claude-sonnet-4-20250514",
                "paper_length": 2500,
                "summary": "This paper introduces the Transformer architecture...",
                "critique": "The summary accurately captures the core contribution...",
                "revised_summary": "This paper introduces the Transformer, a novel neural network architecture...",
                "reflection": "The revision addresses three critical issues identified in the critique...",
                "stage1_metrics": {
                    "timestamp": "2024-11-17T10:30:05Z",
                    "stage": "stage1_summarization",
                    "model": "claude-sonnet-4-20250514",
                    "temperature": 0.3,
                    "input_tokens": 1200,
                    "output_tokens": 800,
                    "total_tokens": 2000,
                    "latency_seconds": 3.5,
                    "stop_reason": "end_turn"
                },
                "total_metrics": {
                    "total_input_tokens": 4500,
                    "total_output_tokens": 3200,
                    "total_tokens": 7700,
                    "total_latency_seconds": 12.3,
                    "stage_count": 3,
                    "average_latency_per_stage": 4.1
                }
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses with detailed diagnostic information."""
    
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details or context")
    timestamp: str = Field(..., description="Timestamp when the error occurred")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Paper text must contain at least 100 characters",
                "detail": "The provided paper text contains only 45 characters",
                "timestamp": "2024-11-17T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check endpoint response."""
    
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current server timestamp")
    components: Dict[str, str] = Field(..., description="Status of individual components")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-11-17T10:30:00Z",
                "components": {
                    "api": "operational",
                    "claude_client": "operational",
                    "monitoring": "operational"
                }
            }
        }


class MonitoringStatsResponse(BaseModel):
    """Schema for monitoring statistics endpoint response."""
    
    total_requests: int = Field(..., description="Total number of requests logged")
    time_window: Dict[str, str] = Field(..., description="Time range of analyzed requests")
    latency: Dict[str, float] = Field(..., description="Latency statistics")
    tokens: Dict[str, Any] = Field(..., description="Token usage statistics")
    user_feedback: Dict[str, Any] = Field(..., description="User feedback statistics")
    stage_breakdown: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Performance breakdown by pipeline stage")


class AnomalyResponse(BaseModel):
    """Schema for detected anomaly information."""
    
    type: str = Field(..., description="Type of anomaly detected")
    severity: str = Field(..., description="Severity level: warning or critical")
    message: str = Field(..., description="Human-readable description of the anomaly")
    metric: float = Field(..., description="Measured metric value that triggered the anomaly")
    threshold: float = Field(..., description="Threshold value that was exceeded")
    timestamp: str = Field(..., description="When the anomaly was detected")