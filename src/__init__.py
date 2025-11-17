"""
Self-Critique Chain Pipeline

A production-ready implementation of the Self-Critique Chain pattern for automated
research paper summarization using Claude AI.
"""

__version__ = "1.0.0"
__author__ = "MLOps Engineer"

from src.pipeline import SelfCritiquePipeline
from src.monitoring import PromptMonitor

__all__ = ["SelfCritiquePipeline", "PromptMonitor"]