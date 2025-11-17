"""
Unit and integration tests for the Self-Critique Chain Pipeline.

This module tests the core pipeline functionality including initialization, stage
execution, error handling, and metrics collection. The tests use mocking to isolate
the pipeline logic from external API dependencies and ensure consistent behavior
across different execution environments.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.pipeline import SelfCritiquePipeline, PipelineError, APIError, ValidationError


class TestSelfCritiquePipeline:
    """Test suite for the SelfCritiquePipeline class."""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client for testing."""
        with patch("src.pipeline.anthropic.Anthropic") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="<output>Test summary</output>")]
            mock_message.usage = MagicMock(
                input_tokens=100,
                output_tokens=50
            )
            mock_message.stop_reason = "end_turn"
            
            mock_instance.messages.create.return_value = mock_message
            
            yield mock_instance
    
    @pytest.fixture
    def pipeline(self, mock_anthropic_client):
        """Create a pipeline instance with mocked client."""
        return SelfCritiquePipeline(api_key="sk-ant-test-key-123456789")
    
    def test_initialization_with_valid_key(self):
        """Test successful pipeline initialization with valid API key."""
        pipeline = SelfCritiquePipeline(api_key="sk-ant-test-key-123456789")
        
        assert pipeline.model == "claude-sonnet-4-20250514"
        assert pipeline.max_tokens == 4096
        assert len(pipeline.metrics) == 0
    
    def test_initialization_with_invalid_key(self):
        """Test that initialization fails with invalid API key format."""
        with pytest.raises(ValidationError, match="Invalid API key format"):
            SelfCritiquePipeline(api_key="invalid-key")
    
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom model and token parameters."""
        pipeline = SelfCritiquePipeline(
            api_key="sk-ant-test-key-123456789",
            model="claude-opus-4",
            max_tokens=8192
        )
        
        assert pipeline.model == "claude-opus-4"
        assert pipeline.max_tokens == 8192
    
    def test_run_pipeline_with_empty_text(self, pipeline):
        """Test that pipeline rejects empty paper text."""
        with pytest.raises(ValidationError, match="non-empty string"):
            pipeline.run_pipeline(paper_text="")
    
    def test_run_pipeline_with_short_text(self, pipeline):
        """Test that pipeline rejects text that is too short."""
        with pytest.raises(ValidationError, match="at least 100 characters"):
            pipeline.run_pipeline(paper_text="Short text")
    
    def test_call_claude_with_success(self, pipeline, mock_anthropic_client):
        """Test successful Claude API call with proper metrics collection."""
        response, metrics = pipeline._call_claude(
            prompt="Test prompt",
            temperature=0.3,
            stage_name="test_stage"
        )
        
        assert response == "<output>Test summary</output>"
        assert metrics["stage"] == "test_stage"
        assert metrics["temperature"] == 0.3
        assert metrics["input_tokens"] == 100
        assert metrics["output_tokens"] == 50
        assert "latency_seconds" in metrics
    
    def test_call_claude_with_api_error(self, pipeline):
        """Test proper error handling when Claude API call fails."""
        with patch.object(pipeline.client.messages, "create") as mock_create:
            mock_create.side_effect = Exception("API connection failed")
            
            with pytest.raises(APIError, match="Unexpected error during API call"):
                pipeline._call_claude(prompt="Test prompt")
    
    def test_metrics_collection(self, pipeline, mock_anthropic_client):
        """Test that metrics are properly collected during execution."""
        initial_count = len(pipeline.metrics)
        
        pipeline._call_claude(prompt="Test prompt", stage_name="test")
        
        assert len(pipeline.metrics) == initial_count + 1
        assert pipeline.metrics[-1]["stage"] == "test"
    
    def test_extract_xml_content_success(self, pipeline):
        """Test successful XML content extraction from response."""
        from src.utils import extract_xml_content
        
        response = "<output>This is the summary</output>"
        content = extract_xml_content(response, "output")
        
        assert content == "This is the summary"
    
    def test_extract_xml_content_missing_tags(self, pipeline):
        """Test XML extraction returns empty string when tags are missing."""
        from src.utils import extract_xml_content
        
        response = "Response without XML tags"
        content = extract_xml_content(response, "output")
        
        assert content == ""
    
    def test_calculate_total_metrics(self, pipeline):
        """Test aggregate metrics calculation across pipeline stages."""
        pipeline.metrics = [
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "latency_seconds": 2.5},
            {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300, "latency_seconds": 3.0},
            {"input_tokens": 150, "output_tokens": 75, "total_tokens": 225, "latency_seconds": 2.8}
        ]
        
        total_metrics = pipeline._calculate_total_metrics()
        
        assert total_metrics["total_input_tokens"] == 450
        assert total_metrics["total_output_tokens"] == 225
        assert total_metrics["total_tokens"] == 675
        assert total_metrics["total_latency_seconds"] == 8.3
        assert total_metrics["stage_count"] == 3


class TestPipelineIntegration:
    """Integration tests for complete pipeline execution."""
    
    @pytest.fixture
    def sample_paper(self):
        """Provide sample paper text for testing."""
        return """
        Title: Attention Is All You Need
        
        Abstract: The dominant sequence transduction models are based on complex 
        recurrent or convolutional neural networks that include an encoder and a 
        decoder. The best performing models also connect the encoder and decoder 
        through an attention mechanism. We propose a new simple network architecture, 
        the Transformer, based solely on attention mechanisms, dispensing with 
        recurrence and convolutions entirely. Experiments on two machine translation 
        tasks show these models to be superior in quality while being more 
        parallelizable and requiring significantly less time to train.
        """
    
    @pytest.fixture
    def mock_stage_responses(self):
        """Provide mock responses for each pipeline stage."""
        return {
            "stage1": "<output>Initial summary of the paper</output><thinking>Planning the summary</thinking>",
            "stage2": "<critique>The summary is accurate but missing key details</critique><thinking>Analyzing the summary</thinking>",
            "stage3": "<output>Revised summary with improvements</output><reflection>Added missing details</reflection><thinking>Revising based on critique</thinking>"
        }
    
    @patch("src.pipeline.anthropic.Anthropic")
    def test_full_pipeline_execution(self, mock_anthropic, sample_paper, mock_stage_responses):
        """Test complete three-stage pipeline execution with all stages."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        stage_counter = [0]
        
        def create_message_side_effect(*args, **kwargs):
            stage_counter[0] += 1
            stage_key = f"stage{stage_counter[0]}"
            
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=mock_stage_responses[stage_key])]
            mock_message.usage = MagicMock(input_tokens=100, output_tokens=50)
            mock_message.stop_reason = "end_turn"
            
            return mock_message
        
        mock_client.messages.create.side_effect = create_message_side_effect
        
        pipeline = SelfCritiquePipeline(api_key="sk-ant-test-key-123456789")
        results = pipeline.run_pipeline(paper_text=sample_paper)
        
        assert "summary" in results
        assert "critique" in results
        assert "revised_summary" in results
        assert "total_metrics" in results
        assert results["total_metrics"]["stage_count"] == 3