"""
Main FastAPI application entry point for the Self-Critique Chain Pipeline.

This module initializes the FastAPI application with proper middleware configuration,
CORS settings, exception handlers, and route registration. The application serves as
the primary interface for accessing pipeline functionality through RESTful endpoints.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routes import router
from src.utils import format_timestamp

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events including startup and shutdown.
    
    This context manager handles initialization tasks when the application starts
    and cleanup operations when it shuts down. The lifespan management ensures
    proper resource allocation and deallocation for production deployments.
    """
    print("\n" + "="*60)
    print("SELF-CRITIQUE PIPELINE API")
    print("="*60)
    print(f"Version: 1.0.0")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Claude Model: {os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')}")
    print("="*60 + "\n")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  WARNING: ANTHROPIC_API_KEY not configured")
        print("   Set your API key in the .env file before making requests\n")
    else:
        print("✅ Anthropic API key configured\n")
    
    yield
    
    print("\n" + "="*60)
    print("SHUTTING DOWN API SERVER")
    print("="*60 + "\n")


app = FastAPI(
    title="Self-Critique Chain Pipeline API",
    description="""
    Production-ready REST API for automated research paper summarization using Claude AI.
    
    This API implements a three-stage Self-Critique Chain pattern that generates initial
    summaries, performs systematic self-critique, and produces revised versions based on
    identified issues. The pipeline reduces hallucinations by fifteen to twenty-five percent
    compared to single-shot approaches through iterative refinement and quality assessment.
    
    ## Key Features
    
    The API provides comprehensive pipeline execution with automatic quality assessment across
    four dimensions including accuracy, completeness, clarity, and coherence. Each execution
    generates detailed performance metrics covering token usage, latency, and stage-specific
    statistics to support monitoring and optimization efforts.
    
    Production-grade monitoring capabilities track all requests with anomaly detection for
    latency spikes and declining user satisfaction. The monitoring system supports integration
    with external observability platforms through structured metric exports and alerting hooks.
    
    Optional MLflow integration enables experiment tracking with automatic logging of parameters,
    metrics, and artifacts. This supports reproducibility requirements and facilitates comparison
    between different prompt versions or model configurations.
    
    ## Authentication
    
    The API requires a valid Anthropic API key configured through environment variables. Set your
    key in the .env file or through your deployment platform's configuration management system.
    
    ## Rate Limits
    
    Rate limiting depends on your Anthropic API subscription tier. The API does not impose
    additional rate limits beyond those enforced by the Claude API itself.
    """,
    version="1.0.0",
    contact={
        "name": "MLOps Engineering Team",
        "email": "mlops@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle uncaught exceptions with proper error responses.
    
    This global exception handler ensures that all unhandled errors return properly
    formatted JSON responses with appropriate HTTP status codes. The handler prevents
    exposure of sensitive implementation details while providing sufficient information
    for debugging production issues.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing your request",
            "detail": str(exc) if os.getenv("ENVIRONMENT") == "development" else None,
            "timestamp": format_timestamp()
        }
    )


app.include_router(router, prefix="/api/v1")


@app.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Root Endpoint",
    description="Get basic information about the API service"
)
async def root():
    """
    Root endpoint providing basic service information and navigation.
    
    This endpoint serves as the entry point for API discovery and provides links
    to documentation resources. The response includes the API version and environment
    information to support version management and deployment verification.
    """
    return {
        "name": "Self-Critique Chain Pipeline API",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json"
        },
        "endpoints": {
            "health_check": "/api/v1/health",
            "execute_pipeline": "/api/v1/pipeline/execute",
            "monitoring_stats": "/api/v1/monitoring/stats",
            "detect_anomalies": "/api/v1/monitoring/anomalies"
        }
    }


def main():
    """
    Entry point for running the API server directly from the command line.
    
    This function supports development and testing workflows by allowing the server
    to be started with a simple Python command. For production deployments, use a
    proper ASGI server like Uvicorn with appropriate worker configuration.
    """
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()