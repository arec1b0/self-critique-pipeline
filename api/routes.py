"""
API route handlers for the Self-Critique Chain Pipeline.

This module implements all HTTP endpoints with proper error handling, logging,
and integration with the core pipeline and monitoring components. The routes
follow RESTful conventions and provide comprehensive OpenAPI documentation.
"""

import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.schemas import (
    PipelineRequest,
    PipelineResponse,
    ErrorResponse,
    HealthResponse,
    MonitoringStatsResponse,
    AnomalyResponse
)
from src.pipeline import SelfCritiquePipeline, PipelineError, APIError, ValidationError
from src.monitoring import PromptMonitor
from src.utils import format_timestamp

router = APIRouter()

monitor = PromptMonitor(
    baseline_latency=float(os.getenv("BASELINE_LATENCY", "2.0")),
    anomaly_threshold_multiplier=2.0,
    satisfaction_threshold=3.0
)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the API service and its components",
    tags=["System"]
)
async def health_check() -> HealthResponse:
    """
    Perform a comprehensive health check of the service and its dependencies.
    
    This endpoint verifies that all critical components are operational including
    the API server, Claude client initialization, and monitoring systems. The health
    check provides immediate visibility into service availability and helps with
    automated monitoring and alerting systems.
    
    Returns:
        HealthResponse containing the overall status and individual component states
    """
    components = {
        "api": "operational",
        "monitoring": "operational"
    }
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        components["claude_client"] = "operational"
    else:
        components["claude_client"] = "not_configured"
    
    overall_status = "healthy" if all(
        v == "operational" for v in components.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=format_timestamp(),
        components=components
    )


@router.post(
    "/pipeline/execute",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Pipeline",
    description="Execute the complete three-stage Self-Critique Chain pipeline on a research paper",
    responses={
        200: {"description": "Pipeline executed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error during pipeline execution"}
    },
    tags=["Pipeline"]
)
async def execute_pipeline(request: PipelineRequest) -> PipelineResponse:
    """
    Execute the complete Self-Critique Chain pipeline on the provided research paper.
    
    This endpoint orchestrates the three-stage process of summarization, critique, and
    revision. The execution includes comprehensive metrics tracking and optional MLflow
    integration for experiment tracking. All requests are logged to the monitoring system
    for performance analysis and anomaly detection.
    
    The pipeline implements a fail-fast strategy where any stage failure immediately
    returns an error response rather than attempting to continue with partial results.
    This ensures that all returned outputs meet the expected quality standards.
    
    Args:
        request: PipelineRequest containing the paper text and configuration parameters
        
    Returns:
        PipelineResponse with all generated content and comprehensive performance metrics
        
    Raises:
        HTTPException: If validation fails, the API key is missing, or pipeline execution encounters errors
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ConfigurationError",
                "message": "Anthropic API key is not configured",
                "timestamp": format_timestamp()
            }
        )
    
    try:
        pipeline = SelfCritiquePipeline(
            api_key=api_key,
            model=request.model,
            max_tokens=4096
        )
        
        results = pipeline.run_pipeline(
            paper_text=request.paper_text,
            mlflow_tracking=request.enable_mlflow,
            experiment_name=request.experiment_name
        )
        
        for stage_num in range(1, 4):
            stage_metrics = results.get(f"stage{stage_num}_metrics", {})
            if stage_metrics:
                monitor.log_request(
                    prompt=f"Stage {stage_num} prompt",
                    response=results.get("summary" if stage_num == 1 else "critique" if stage_num == 2 else "revised_summary", ""),
                    metrics=stage_metrics,
                    user_feedback=None,
                    metadata={"stage": stage_num, "model": request.model}
                )
        
        response_data = {
            "pipeline_start": results["pipeline_start"],
            "pipeline_end": results["pipeline_end"],
            "model": results["model"],
            "paper_length": results["paper_length"],
            "summary": results["summary"],
            "critique": results["critique"],
            "revised_summary": results["revised_summary"],
            "reflection": results["reflection"],
            "stage1_metrics": results["stage1_metrics"],
            "stage2_metrics": results["stage2_metrics"],
            "stage3_metrics": results["stage3_metrics"],
            "total_metrics": results["total_metrics"]
        }
        
        return PipelineResponse(**response_data)
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": str(e),
                "timestamp": format_timestamp()
            }
        )
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "APIError",
                "message": str(e),
                "detail": "Failed to communicate with Claude API. Check your API key and network connectivity.",
                "timestamp": format_timestamp()
            }
        )
    except PipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PipelineError",
                "message": str(e),
                "timestamp": format_timestamp()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred during pipeline execution",
                "detail": str(e),
                "timestamp": format_timestamp()
            }
        )


@router.get(
    "/monitoring/stats",
    response_model=MonitoringStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Monitoring Statistics",
    description="Retrieve comprehensive statistics about pipeline performance and resource usage",
    tags=["Monitoring"]
)
async def get_monitoring_stats(window_size: int = None) -> MonitoringStatsResponse:
    """
    Retrieve comprehensive monitoring statistics for pipeline performance analysis.
    
    This endpoint provides aggregate metrics across all logged requests including
    latency distributions, token usage patterns, user feedback scores, and performance
    breakdowns by pipeline stage. The statistics support operational monitoring and
    capacity planning decisions.
    
    Args:
        window_size: Optional number of recent requests to analyze. If not provided, analyzes all requests.
        
    Returns:
        MonitoringStatsResponse containing comprehensive performance statistics
    """
    stats = monitor.get_summary_stats(window_size=window_size)
    return MonitoringStatsResponse(**stats)


@router.get(
    "/monitoring/anomalies",
    response_model=list[AnomalyResponse],
    status_code=status.HTTP_200_OK,
    summary="Detect Anomalies",
    description="Detect performance anomalies in recent pipeline executions",
    tags=["Monitoring"]
)
async def detect_anomalies(window_size: int = 100) -> list[AnomalyResponse]:
    """
    Detect performance anomalies in recent pipeline executions.
    
    This endpoint analyzes recent requests to identify patterns that indicate performance
    degradation or quality issues. Detected anomalies include latency spikes, declining
    user satisfaction, and excessive token consumption. The results help identify issues
    before they impact a significant number of users.
    
    Args:
        window_size: Number of recent requests to analyze for anomaly detection
        
    Returns:
        List of detected anomalies with severity levels and diagnostic information
    """
    anomalies = monitor.detect_anomalies(window_size=window_size)
    
    if anomalies:
        monitor.trigger_alert(anomalies)
    
    return [AnomalyResponse(**anomaly) for anomaly in anomalies]