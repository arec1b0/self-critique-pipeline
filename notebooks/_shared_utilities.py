"""
Shared utilities for Self-Critique Pipeline notebooks.

This module provides reusable functions across all notebooks including
data loading, metric calculation, visualization, and reporting utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring import PromptMonitor
from src.pipeline import SelfCritiquePipeline


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_monitoring_data(filepath: Optional[str] = None, 
                         window_days: int = 7) -> pd.DataFrame:
    """
    Load monitoring data from logs.
    
    Args:
        filepath: Path to log file. If None, loads from default location.
        window_days: Number of days to load data for.
        
    Returns:
        DataFrame with monitoring metrics
    """
    if filepath is None:
        log_dir = project_root / "logs"
        log_files = sorted(log_dir.glob("pipeline_logs_*.json"))
        if not log_files:
            return pd.DataFrame()
        filepath = log_files[-1]
    
    with open(filepath, 'r') as f:
        logs = json.load(f)
    
    df = pd.DataFrame(logs)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = datetime.now() - timedelta(days=window_days)
        df = df[df['timestamp'] >= cutoff]
    
    return df


def create_benchmark_dataset() -> List[Dict[str, str]]:
    """
    Create a benchmark dataset of diverse research papers.
    
    Returns:
        List of paper dictionaries with 'title', 'text', and 'category'
    """
    papers = [
        {
            "title": "Attention Is All You Need",
            "category": "deep-learning",
            "text": """Title: Attention Is All You Need

Abstract: The dominant sequence transduction models are based on complex 
recurrent or convolutional neural networks that include an encoder and decoder. 
The best performing models also connect the encoder and decoder through an 
attention mechanism. We propose a new simple network architecture, the 
Transformer, based solely on attention mechanisms, dispensing with recurrence 
and convolutions entirely."""
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "category": "nlp",
            "text": """Title: BERT: Pre-training of Deep Bidirectional Transformers for 
Language Understanding

Abstract: We introduce a new language representation model called BERT, which 
stands for Bidirectional Encoder Representations from Transformers. Unlike 
recent language representation models, BERT is designed to pre-train deep 
bidirectional representations from unlabeled text by jointly conditioning on 
both left and right context in all layers."""
        },
        {
            "title": "ResNet: Deep Residual Learning",
            "category": "computer-vision",
            "text": """Title: Deep Residual Learning for Image Recognition

Abstract: Deeper neural networks are more difficult to train. We present a 
residual learning framework to ease the training of networks that are 
substantially deeper than those used previously. We explicitly reformulate 
the layers as learning residual functions with reference to the layer inputs, 
instead of learning unreferenced functions."""
        }
    ]
    
    return papers


# ============================================================================
# Metric Calculation Functions
# ============================================================================

def calculate_cost_metrics(results: Dict[str, Any], 
                           model: str = "claude-sonnet-4-20250514") -> Dict[str, float]:
    """
    Calculate cost metrics for pipeline execution.
    
    Args:
        results: Pipeline execution results
        model: Model name for pricing lookup
        
    Returns:
        Dictionary with cost breakdowns
    """
    # Pricing per 1M tokens (adjust based on current Anthropic pricing)
    pricing = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-haiku-4-20250514": {"input": 0.80, "output": 4.00}
    }
    
    if model not in pricing:
        model = "claude-sonnet-4-20250514"
    
    total_metrics = results.get("total_metrics", {})
    input_tokens = total_metrics.get("total_input_tokens", 0)
    output_tokens = total_metrics.get("total_output_tokens", 0)
    
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_1k_tokens": round((total_cost / (input_tokens + output_tokens)) * 1000, 4)
    }


def calculate_quality_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract quality metrics from pipeline results.
    
    Args:
        results: Pipeline execution results
        
    Returns:
        Dictionary with quality scores
    """
    critique = results.get("critique", "")
    
    # Parse quality scores from critique
    scores = {
        "accuracy": 0.0,
        "completeness": 0.0,
        "clarity": 0.0,
        "coherence": 0.0,
        "overall": 0.0
    }
    
    # Simple parsing logic - extract scores from critique text
    for dimension in scores.keys():
        if f"**{dimension.capitalize()}:**" in critique.lower():
            # Extract score (assumes format like "Accuracy: 8/10")
            try:
                start = critique.lower().find(f"{dimension}:")
                end = critique.find("\n", start)
                score_text = critique[start:end]
                # Extract number
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                if match:
                    scores[dimension] = float(match.group(1))
            except:
                pass
    
    return scores


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_performance_graphs(df: pd.DataFrame, 
                            metric: str = "latency_seconds",
                            title: Optional[str] = None) -> None:
    """
    Plot performance metrics over time.
    
    Args:
        df: DataFrame with monitoring data
        metric: Metric column to plot
        title: Optional plot title
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series
    if 'timestamp' in df.columns:
        df.set_index('timestamp')[metric].plot(ax=axes[0, 0])
        axes[0, 0].set_title(f'{metric} Over Time')
        axes[0, 0].set_ylabel(metric)
    
    # Distribution
    df[metric].hist(bins=30, ax=axes[0, 1])
    axes[0, 1].set_title(f'{metric} Distribution')
    axes[0, 1].set_xlabel(metric)
    
    # By stage
    if 'stage' in df.columns:
        df.groupby('stage')[metric].mean().plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title(f'Average {metric} by Stage')
        axes[1, 0].set_ylabel(metric)
    
    # Box plot
    if 'stage' in df.columns:
        df.boxplot(column=metric, by='stage', ax=axes[1, 1])
        axes[1, 1].set_title(f'{metric} Distribution by Stage')
    
    plt.tight_layout()
    if title:
        fig.suptitle(title, y=1.02)
    plt.show()


def plot_cost_breakdown(cost_data: List[Dict[str, Any]]) -> None:
    """
    Visualize cost breakdown across executions.
    
    Args:
        cost_data: List of cost metric dictionaries
    """
    df = pd.DataFrame(cost_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total cost trend
    if len(df) > 1:
        df['total_cost_usd'].plot(ax=axes[0], marker='o')
        axes[0].set_title('Cost per Execution')
        axes[0].set_ylabel('Cost (USD)')
        axes[0].set_xlabel('Execution #')
    
    # Token distribution
    df[['input_tokens', 'output_tokens']].plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Token Distribution')
    axes[1].set_ylabel('Tokens')
    axes[1].legend(['Input', 'Output'])
    
    # Cost breakdown
    df[['input_cost_usd', 'output_cost_usd']].mean().plot(kind='pie', ax=axes[2], autopct='%1.1f%%')
    axes[2].set_title('Average Cost Breakdown')
    axes[2].set_ylabel('')
    
    plt.tight_layout()
    plt.show()


def plot_quality_comparison(quality_data: List[Dict[str, float]]) -> None:
    """
    Visualize quality metrics comparison.
    
    Args:
        quality_data: List of quality metric dictionaries
    """
    df = pd.DataFrame(quality_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Radar chart for dimensions
    categories = ['accuracy', 'completeness', 'clarity', 'coherence']
    if all(cat in df.columns for cat in categories):
        means = df[categories].mean()
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        means_list = means.tolist()
        means_list += means_list[:1]
        angles += angles[:1]
        
        ax = plt.subplot(121, polar=True)
        ax.plot(angles, means_list, 'o-', linewidth=2)
        ax.fill(angles, means_list, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_ylim(0, 10)
        ax.set_title('Average Quality Dimensions')
    
    # Overall score distribution
    if 'overall' in df.columns:
        df['overall'].hist(bins=20, ax=axes[1])
        axes[1].set_title('Overall Quality Score Distribution')
        axes[1].set_xlabel('Score')
        axes[1].axvline(df['overall'].mean(), color='red', linestyle='--', label='Mean')
        axes[1].legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Reporting Functions
# ============================================================================

def export_to_executive_summary(results: Dict[str, Any], 
                                cost_metrics: Dict[str, float],
                                quality_metrics: Dict[str, float],
                                output_path: str = "executive_summary.json") -> None:
    """
    Export executive summary with key metrics.
    
    Args:
        results: Pipeline results
        cost_metrics: Cost breakdown
        quality_metrics: Quality scores
        output_path: Output file path
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "performance": {
            "total_latency_seconds": results.get("total_metrics", {}).get("total_latency_seconds", 0),
            "average_latency_per_stage": results.get("total_metrics", {}).get("average_latency_per_stage", 0)
        },
        "cost": cost_metrics,
        "quality": quality_metrics,
        "model": results.get("model", "unknown"),
        "paper_length": results.get("paper_length", 0)
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Executive summary exported to {output_path}")


def setup_mlflow_context(experiment_name: str = "self-critique-pipeline") -> None:
    """
    Setup MLflow tracking context.
    
    Args:
        experiment_name: Name of MLflow experiment
    """
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: {experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    except ImportError:
        print("Warning: MLflow not installed. Install with: pip install mlflow")


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def calculate_percentiles(data: List[float], 
                          percentiles: List[int] = [50, 95, 99]) -> Dict[str, float]:
    """
    Calculate percentile values for performance analysis.
    
    Args:
        data: List of numeric values
        percentiles: List of percentile values to calculate
        
    Returns:
        Dictionary mapping percentile to value
    """
    return {f"p{p}": np.percentile(data, p) for p in percentiles}


def detect_outliers(data: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Detect outliers in data using specified method.
    
    Args:
        data: Series of numeric values
        method: 'iqr' for interquartile range or 'zscore' for z-score method
        
    Returns:
        Boolean series indicating outliers
    """
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > 3
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_distributions(data1: pd.Series, 
                         data2: pd.Series,
                         test: str = "ks") -> Tuple[float, float]:
    """
    Compare two distributions using statistical tests.
    
    Args:
        data1: First data series
        data2: Second data series
        test: 'ks' for Kolmogorov-Smirnov or 't' for t-test
        
    Returns:
        Tuple of (statistic, p_value)
    """
    from scipy import stats
    
    if test == "ks":
        return stats.ks_2samp(data1, data2)
    elif test == "t":
        return stats.ttest_ind(data1, data2)
    else:
        raise ValueError(f"Unknown test: {test}")


# ============================================================================
# Helper Functions
# ============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"


def format_cost(usd: float) -> str:
    """Format cost in human-readable format."""
    if usd < 0.01:
        return f"${usd*100:.2f}¢"
    else:
        return f"${usd:.4f}"


def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics Summary") -> None:
    """Print metrics in formatted table."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'cost' in key.lower():
                print(f"  {key:.<40} {format_cost(value):>15}")
            elif 'latency' in key.lower() or 'seconds' in key.lower():
                print(f"  {key:.<40} {format_duration(value):>15}")
            else:
                print(f"  {key:.<40} {value:>15.2f}")
        else:
            print(f"  {key:.<40} {str(value):>15}")
    
    print(f"{'='*60}\n")

