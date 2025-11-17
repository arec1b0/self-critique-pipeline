"""
Monitoring and observability module for the Self-Critique Chain Pipeline.

This module provides production-grade monitoring capabilities including request logging,
metrics calculation, anomaly detection, and alert triggering. The implementation follows
the Single Responsibility Principle with clear separation between logging, analysis, and
alerting concerns.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from src.utils import format_timestamp


class MonitoringError(Exception):
    """Base exception for monitoring-related errors."""
    pass


class PromptMonitor:
    """
    Monitor for tracking Claude API request performance and detecting anomalies.
    
    This class provides comprehensive observability for the pipeline by logging all
    requests with detailed metrics including token usage, latency, and user feedback.
    The monitor calculates aggregate statistics and detects anomalies such as latency
    spikes or declining user satisfaction scores.
    
    The implementation enables production-grade monitoring with support for exporting
    logs to various formats for integration with external observability platforms like
    Prometheus, Grafana, or custom dashboards.
    
    Attributes:
        baseline_latency: Expected baseline latency in seconds for anomaly detection
        logs: List of all logged requests with their associated metrics
        anomaly_threshold_multiplier: Factor for latency anomaly detection
        satisfaction_threshold: Minimum acceptable user satisfaction score
    """
    
    def __init__(
        self,
        baseline_latency: float = 2.0,
        anomaly_threshold_multiplier: float = 2.0,
        satisfaction_threshold: float = 3.0
    ):
        """
        Initialize the monitoring system with configurable thresholds.
        
        Args:
            baseline_latency: Expected baseline latency in seconds
            anomaly_threshold_multiplier: Multiplier for latency anomaly detection
            satisfaction_threshold: Minimum acceptable satisfaction score on 1-5 scale
        """
        self.baseline_latency = baseline_latency
        self.anomaly_threshold_multiplier = anomaly_threshold_multiplier
        self.satisfaction_threshold = satisfaction_threshold
        self.logs = []
    
    def log_request(
        self,
        prompt: str,
        response: str,
        metrics: Dict[str, Any],
        user_feedback: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a Claude API request with comprehensive metrics and context.
        
        This method captures all relevant information about a request including the
        input prompt, generated response, performance metrics, and optional user
        feedback. The logged data supports both real-time monitoring and historical
        analysis for identifying trends or degradation patterns.
        
        Args:
            prompt: Input prompt sent to Claude
            response: Generated response from Claude
            metrics: Dictionary containing performance metrics such as tokens and latency
            user_feedback: Optional user satisfaction score on a scale from one to five
            metadata: Optional additional context or tags for categorization
        """
        log_entry = {
            "timestamp": format_timestamp(),
            "prompt_length": len(prompt),
            "response_length": len(response),
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "total_tokens": metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0),
            "latency_seconds": metrics.get("latency_seconds", 0),
            "model": metrics.get("model", "unknown"),
            "temperature": metrics.get("temperature", 0),
            "stage": metrics.get("stage", "unknown"),
            "user_feedback": user_feedback,
            "metadata": metadata or {}
        }
        
        self.logs.append(log_entry)
    
    def detect_anomalies(self, window_size: int = 100) -> List[Dict[str, Any]]:
        """
        Detect anomalies in recent request patterns.
        
        This method analyzes the most recent requests within the specified window to
        identify performance degradation or quality issues. Anomalies are detected
        across multiple dimensions including latency spikes and declining user
        satisfaction scores.
        
        The detection algorithm compares current performance against established
        baselines and triggers alerts when thresholds are exceeded. This enables
        proactive intervention before issues impact a large number of users.
        
        Args:
            window_size: Number of recent requests to analyze for anomaly detection
            
        Returns:
            List of detected anomalies with severity levels and diagnostic information
        """
        if not self.logs:
            return []
        
        anomalies = []
        recent_logs = self.logs[-window_size:]
        
        latencies = [log["latency_seconds"] for log in recent_logs]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        if avg_latency > self.baseline_latency * self.anomaly_threshold_multiplier:
            anomalies.append({
                "type": "high_average_latency",
                "severity": "warning",
                "message": (
                    f"Average latency of {avg_latency:.2f} seconds exceeds "
                    f"{self.anomaly_threshold_multiplier}x baseline of "
                    f"{self.baseline_latency} seconds"
                ),
                "metric": avg_latency,
                "threshold": self.baseline_latency * self.anomaly_threshold_multiplier,
                "window_size": len(recent_logs),
                "timestamp": format_timestamp()
            })
        
        if max_latency > self.baseline_latency * self.anomaly_threshold_multiplier * 1.5:
            anomalies.append({
                "type": "latency_spike",
                "severity": "critical",
                "message": (
                    f"Maximum latency spike of {max_latency:.2f} seconds detected "
                    f"which is significantly above acceptable thresholds"
                ),
                "metric": max_latency,
                "threshold": self.baseline_latency * self.anomaly_threshold_multiplier * 1.5,
                "window_size": len(recent_logs),
                "timestamp": format_timestamp()
            })
        
        feedback_logs = [log for log in recent_logs if log["user_feedback"] is not None]
        if feedback_logs:
            feedback_scores = [log["user_feedback"] for log in feedback_logs]
            avg_feedback = sum(feedback_scores) / len(feedback_scores)
            
            if avg_feedback < self.satisfaction_threshold:
                anomalies.append({
                    "type": "low_user_satisfaction",
                    "severity": "critical",
                    "message": (
                        f"Average user satisfaction of {avg_feedback:.2f} out of 5 "
                        f"falls below the minimum threshold of {self.satisfaction_threshold}"
                    ),
                    "metric": avg_feedback,
                    "threshold": self.satisfaction_threshold,
                    "sample_size": len(feedback_logs),
                    "timestamp": format_timestamp()
                })
        
        token_counts = [log["total_tokens"] for log in recent_logs]
        avg_tokens = sum(token_counts) / len(token_counts)
        
        if avg_tokens > 8000:
            anomalies.append({
                "type": "high_token_usage",
                "severity": "warning",
                "message": (
                    f"Average token usage of {avg_tokens:.0f} tokens per request "
                    f"may indicate inefficient prompting or excessive context"
                ),
                "metric": avg_tokens,
                "threshold": 8000,
                "window_size": len(recent_logs),
                "timestamp": format_timestamp()
            })
        
        return anomalies
    
    def get_summary_stats(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate summary statistics for logged requests.
        
        This method computes aggregate metrics across all logged requests or within
        a specified window. The statistics provide insights into overall system
        performance and resource utilization patterns.
        
        Args:
            window_size: Optional number of recent requests to analyze. If None,
                        analyzes all logged requests.
            
        Returns:
            Dictionary containing comprehensive summary statistics including averages,
            totals, and distributions across various performance dimensions
        """
        if not self.logs:
            return {
                "total_requests": 0,
                "error": "No requests have been logged yet"
            }
        
        logs_to_analyze = self.logs[-window_size:] if window_size else self.logs
        
        latencies = [log["latency_seconds"] for log in logs_to_analyze]
        token_counts = [log["total_tokens"] for log in logs_to_analyze]
        
        feedback_logs = [log for log in logs_to_analyze if log["user_feedback"] is not None]
        feedback_scores = [log["user_feedback"] for log in feedback_logs] if feedback_logs else []
        
        stats = {
            "total_requests": len(logs_to_analyze),
            "time_window": {
                "start": logs_to_analyze[0]["timestamp"],
                "end": logs_to_analyze[-1]["timestamp"]
            },
            "latency": {
                "average_seconds": round(sum(latencies) / len(latencies), 3),
                "min_seconds": round(min(latencies), 3),
                "max_seconds": round(max(latencies), 3),
                "median_seconds": round(sorted(latencies)[len(latencies) // 2], 3)
            },
            "tokens": {
                "average_per_request": round(sum(token_counts) / len(token_counts), 1),
                "total_consumed": sum(token_counts),
                "min_per_request": min(token_counts),
                "max_per_request": max(token_counts)
            },
            "user_feedback": {
                "sample_size": len(feedback_scores),
                "average_score": round(sum(feedback_scores) / len(feedback_scores), 2) if feedback_scores else None,
                "distribution": self._calculate_feedback_distribution(feedback_scores) if feedback_scores else None
            }
        }
        
        stage_breakdown = self._calculate_stage_breakdown(logs_to_analyze)
        if stage_breakdown:
            stats["stage_breakdown"] = stage_breakdown
        
        return stats
    
    def _calculate_feedback_distribution(self, scores: List[int]) -> Dict[int, int]:
        """Calculate distribution of user feedback scores."""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for score in scores:
            if 1 <= score <= 5:
                distribution[score] += 1
        return distribution
    
    def _calculate_stage_breakdown(self, logs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics broken down by pipeline stage."""
        stages = {}
        
        for log in logs:
            stage = log.get("stage", "unknown")
            if stage not in stages:
                stages[stage] = {
                    "count": 0,
                    "total_latency": 0,
                    "total_tokens": 0
                }
            
            stages[stage]["count"] += 1
            stages[stage]["total_latency"] += log["latency_seconds"]
            stages[stage]["total_tokens"] += log["total_tokens"]
        
        for stage, data in stages.items():
            data["average_latency"] = round(data["total_latency"] / data["count"], 3)
            data["average_tokens"] = round(data["total_tokens"] / data["count"], 1)
        
        return stages
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export logs to a pandas DataFrame for advanced analysis.
        
        This method converts the collected logs into a structured DataFrame format
        that supports complex queries, visualizations, and integration with data
        analysis tools. The DataFrame includes all logged metrics and can be easily
        filtered, grouped, or aggregated for specific analytical needs.
        
        Returns:
            DataFrame containing all logged requests with their associated metrics
        """
        if not self.logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export logs to a JSON file for persistence or external processing.
        
        Args:
            filepath: Path where the JSON file should be written
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def clear_logs(self) -> int:
        """
        Clear all logged requests from memory.
        
        Returns:
            Number of log entries that were cleared
        """
        count = len(self.logs)
        self.logs = []
        return count
    
    def trigger_alert(self, anomalies: List[Dict[str, Any]]) -> None:
        """
        Trigger alerts for detected anomalies.
        
        This method serves as an integration point for external alerting systems
        such as Slack webhooks, PagerDuty, or email notifications. In production
        deployments, this should be extended to actually send alerts through the
        appropriate channels based on anomaly severity.
        
        Args:
            anomalies: List of detected anomalies to alert on
        """
        if not anomalies:
            return
        
        critical_anomalies = [a for a in anomalies if a["severity"] == "critical"]
        warning_anomalies = [a for a in anomalies if a["severity"] == "warning"]
        
        if critical_anomalies:
            print("\n" + "="*60)
            print("🚨 CRITICAL ALERTS")
            print("="*60)
            for anomaly in critical_anomalies:
                print(f"\n[{anomaly['type'].upper()}]")
                print(f"  {anomaly['message']}")
                print(f"  Detected at: {anomaly['timestamp']}")
        
        if warning_anomalies:
            print("\n" + "="*60)
            print("⚠️  WARNING ALERTS")
            print("="*60)
            for anomaly in warning_anomalies:
                print(f"\n[{anomaly['type'].upper()}]")
                print(f"  {anomaly['message']}")
                print(f"  Detected at: {anomaly['timestamp']}")
        
        print("\n" + "="*60 + "\n")