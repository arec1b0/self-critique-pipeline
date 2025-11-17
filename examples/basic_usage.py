"""
Basic usage example for the Self-Critique Chain Pipeline.

This script demonstrates the fundamental workflow for executing the pipeline
on a research paper with minimal configuration. The example shows how to
initialize the pipeline, execute it on sample content, and access the results.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import SelfCritiquePipeline

def main():
    """Execute basic pipeline demonstration."""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        print("Please set your API key in the .env file")
        return
    
    sample_paper = """
    Title: Attention Is All You Need
    
    Abstract: The dominant sequence transduction models are based on complex 
    recurrent or convolutional neural networks that include an encoder and a 
    decoder. We propose a new simple network architecture, the Transformer, 
    based solely on attention mechanisms, dispensing with recurrence and 
    convolutions entirely. Experiments on two machine translation tasks show 
    these models to be superior in quality while being more parallelizable 
    and requiring significantly less time to train.
    """
    
    print("\n" + "="*60)
    print("SELF-CRITIQUE CHAIN PIPELINE - BASIC USAGE EXAMPLE")
    print("="*60 + "\n")
    
    pipeline = SelfCritiquePipeline(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        max_tokens=4096
    )
    
    print("Executing pipeline on sample paper...\n")
    
    results = pipeline.run_pipeline(
        paper_text=sample_paper,
        mlflow_tracking=False
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    print("INITIAL SUMMARY:")
    print("-" * 60)
    print(results["summary"])
    
    print("\n\nREVISED SUMMARY:")
    print("-" * 60)
    print(results["revised_summary"])
    
    print("\n\nPERFORMANCE METRICS:")
    print("-" * 60)
    total_metrics = results["total_metrics"]
    print(f"Total tokens consumed: {total_metrics['total_tokens']}")
    print(f"Total latency: {total_metrics['total_latency_seconds']:.2f} seconds")
    print(f"Stages completed: {total_metrics['stage_count']}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()