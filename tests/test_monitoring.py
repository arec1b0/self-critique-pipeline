"""
Unit tests for the monitoring and observability module.

This module tests the monitoring system functionality including request logging,
metrics calculation, anomaly detection algorithms, and data export capabilities.
The tests ensure that the monitoring system correctly tracks performance and
identifies issues that require attention.
"""

import pytest
import pandas as pd
from src.monitoring import PromptMonitor, MonitoringError


class TestPromptMonitor:
    """Test suite for the PromptMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor instance for each test."""
        return PromptMonitor(
            baseline_latency=2.0,
            anomaly_threshold_multiplier=2.0,
            satisfaction_threshold=3.0
        )
    
    @pytest.fixture
    def sample_metrics(self):
        """Provide sample metrics for testing."""
        return {
            "input_tokens": 100,
            "output_tokens": 50,
            "latency_seconds": 1.5,
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "stage": "stage1_summarization"
        }
    
    def test_initialization(self, monitor):
        """Test monitor initializes with correct default values."""
        assert monitor.baseline_latency == 2.0
        assert monitor.anomaly_threshold_multiplier == 2.0
        assert monitor.satisfaction_threshold == 3.0
        assert len(monitor.logs) == 0
    
    def test_log_request_basic(self, monitor, sample_metrics):
        """Test basic request logging functionality."""
        monitor.log_request(
            prompt="Test prompt",
            response="Test response",
            metrics=sample_metrics
        )
        
        assert len(monitor.logs) == 1
        assert monitor.logs[0]["prompt_length"] == len("Test prompt")
        assert monitor.logs[0]["response_length"] == len("Test response")
        assert monitor.logs[0]["input_tokens"] == 100
        assert monitor.logs[0]["output_tokens"] == 50
    
    def test_log_request_with_feedback(self, monitor, sample_metrics):
        """Test logging with user feedback scores."""
        monitor.log_request(
            prompt="Test prompt",
            response="Test response",
            metrics=sample_metrics,
            user_feedback=4
        )
        
        assert monitor.logs[0]["user_feedback"] == 4
    
    def test_log_request_with_metadata(self, monitor, sample_metrics):
        """Test logging with additional metadata."""
        metadata = {"experiment": "test-run", "version": "1.0"}
        
        monitor.log_request(
            prompt="Test prompt",
            response="Test response",
            metrics=sample_metrics,
            metadata=metadata
        )
        
        assert monitor.logs[0]["metadata"]["experiment"] == "test-run"
    
    def test_detect_anomalies_no_logs(self, monitor):
        """Test anomaly detection returns empty list when no logs exist."""
        anomalies = monitor.detect_anomalies()
        assert len(anomalies) == 0
    
    def test_detect_high_latency_anomaly(self, monitor, sample_metrics):
        """Test detection of high latency anomaly."""
        high_latency_metrics = sample_metrics.copy()
        high_latency_metrics["latency_seconds"] = 5.0
        
        for _ in range(10):
            monitor.log_request("prompt", "response", high_latency_metrics)
        
        anomalies = monitor.detect_anomalies(window_size=10)
        
        anomaly_types = [a["type"] for a in anomalies]
        assert "high_average_latency" in anomaly_types
    
    def test_detect_latency_spike_anomaly(self, monitor, sample_metrics):
        """Test detection of extreme latency spike."""
        normal_metrics = sample_metrics.copy()
        normal_metrics["latency_seconds"] = 1.0
        
        for _ in range(9):
            monitor.log_request("prompt", "response", normal_metrics)
        
        spike_metrics = sample_metrics.copy()
        spike_metrics["latency_seconds"] = 10.0
        monitor.log_request("prompt", "response", spike_metrics)
        
        anomalies = monitor.detect_anomalies(window_size=10)
        
        anomaly_types = [a["type"] for a in anomalies]
        assert "latency_spike" in anomaly_types
    
    def test_detect_low_satisfaction_anomaly(self, monitor, sample_metrics):
        """Test detection of low user satisfaction scores."""
        for score in [2, 2, 1, 2, 1]:
            monitor.log_request("prompt", "response", sample_metrics, user_feedback=score)
        
        anomalies = monitor.detect_anomalies(window_size=5)
        
        anomaly_types = [a["type"] for a in anomalies]
        assert "low_user_satisfaction" in anomaly_types
    
    def test_detect_high_token_usage_anomaly(self, monitor, sample_metrics):
        """Test detection of excessive token usage."""
        high_token_metrics = sample_metrics.copy()
        high_token_metrics["input_tokens"] = 5000
        high_token_metrics["output_tokens"] = 4000
        
        for _ in range(10):
            monitor.log_request("prompt", "response", high_token_metrics)
        
        anomalies = monitor.detect_anomalies(window_size=10)
        
        anomaly_types = [a["type"] for a in anomalies]
        assert "high_token_usage" in anomaly_types
    
    def test_get_summary_stats_empty(self, monitor):
        """Test summary statistics with no logged requests."""
        stats = monitor.get_summary_stats()
        
        assert stats["total_requests"] == 0
        assert "error" in stats
    
    def test_get_summary_stats_with_data(self, monitor, sample_metrics):
        """Test summary statistics calculation with logged data."""
        for i in range(5):
            metrics = sample_metrics.copy()
            metrics["latency_seconds"] = 1.0 + i * 0.5
            monitor.log_request("prompt", "response", metrics)
        
        stats = monitor.get_summary_stats()
        
        assert stats["total_requests"] == 5
        assert "latency" in stats
        assert "tokens" in stats
        assert stats["latency"]["min_seconds"] == 1.0
        assert stats["latency"]["max_seconds"] == 3.0
    
    def test_get_summary_stats_with_window(self, monitor, sample_metrics):
        """Test summary statistics with window size limitation."""
        for _ in range(20):
            monitor.log_request("prompt", "response", sample_metrics)
        
        stats = monitor.get_summary_stats(window_size=10)
        
        assert stats["total_requests"] == 10
    
    def test_export_to_dataframe_empty(self, monitor):
        """Test DataFrame export with no data returns empty DataFrame."""
        df = monitor.export_to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_export_to_dataframe_with_data(self, monitor, sample_metrics):
        """Test DataFrame export with logged data."""
        for _ in range(5):
            monitor.log_request("prompt", "response", sample_metrics)
        
        df = monitor.export_to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "input_tokens" in df.columns
        assert "latency_seconds" in df.columns
    
    def test_clear_logs(self, monitor, sample_metrics):
        """Test clearing logged requests."""
        for _ in range(10):
            monitor.log_request("prompt", "response", sample_metrics)
        
        assert len(monitor.logs) == 10
        
        cleared_count = monitor.clear_logs()
        
        assert cleared_count == 10
        assert len(monitor.logs) == 0