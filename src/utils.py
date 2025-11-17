"""
Utility functions for the Self-Critique Chain Pipeline.

This module provides helper functions for XML content extraction, token counting,
and other common operations used throughout the pipeline.
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime


def extract_xml_content(response: str, tag: str) -> str:
    """
    Extract content between XML tags from Claude response.
    
    This function parses Claude's structured XML output to extract specific sections
    such as thinking, output, critique, or reflection content. The extraction is
    case-sensitive and returns an empty string if the tags are not found.
    
    Args:
        response: Full Claude API response text containing XML tags
        tag: XML tag name to extract (e.g., 'thinking', 'output', 'critique')
        
    Returns:
        Extracted content between the specified tags, or empty string if not found
        
    Examples:
        >>> response = "<output>Summary text here</output>"
        >>> extract_xml_content(response, "output")
        'Summary text here'
    """
    start = response.find(f"<{tag}>")
    end = response.find(f"</{tag}>")
    
    if start == -1 or end == -1:
        return ""
    
    return response[start + len(tag) + 2:end].strip()


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count for text content.
    
    This function provides a rough estimation of token count using a simple heuristic
    of 4 characters per token. For production use with precise billing requirements,
    consider using the tiktoken library for exact counts.
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Approximate number of tokens
        
    Note:
        This is an approximation. Actual token counts may vary based on the
        tokenizer used by the Claude model.
    """
    return len(text) // 4


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime object to ISO 8601 string.
    
    Args:
        dt: Datetime object to format. If None, uses current UTC time.
        
    Returns:
        ISO 8601 formatted timestamp string
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + "Z"


def validate_api_key(api_key: str) -> bool:
    """
    Validate Anthropic API key format.
    
    This function performs basic format validation on the API key to catch
    common configuration errors before making API calls.
    
    Args:
        api_key: Anthropic API key to validate
        
    Returns:
        True if the key format appears valid, False otherwise
        
    Note:
        This only validates format, not whether the key is active or has
        sufficient credits.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Anthropic API keys typically start with 'sk-ant-'
    if not api_key.startswith("sk-ant-"):
        return False
    
    # Keys should be of reasonable length
    if len(api_key) < 20:
        return False
    
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove invalid characters.
    
    This function removes or replaces characters that are invalid in Windows
    filenames while preserving readability.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for Windows filesystem
    """
    # Remove invalid Windows filename characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, "_", filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")
    
    # Limit length to reasonable value
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def parse_self_assessment(critique_text: str) -> Dict[str, float]:
    """
    Parse self-assessment scores from critique output.
    
    This function extracts numerical scores for accuracy, completeness, clarity,
    coherence, and overall quality from the structured critique text.
    
    Args:
        critique_text: Critique section containing self-assessment scores
        
    Returns:
        Dictionary mapping dimension names to scores (0-10 scale)
        
    Examples:
        >>> critique = "**Accuracy:** 8.5 - Well preserved\\n**Completeness:** 7.0 - Missing details"
        >>> parse_self_assessment(critique)
        {'accuracy': 8.5, 'completeness': 7.0}
    """
    scores = {}
    
    dimensions = ["accuracy", "completeness", "clarity", "coherence", "overall quality"]
    
    for dimension in dimensions:
        pattern = rf"\*\*{dimension.title()}:\*\*\s*(\d+\.?\d*)"
        match = re.search(pattern, critique_text, re.IGNORECASE)
        
        if match:
            scores[dimension.lower().replace(" ", "_")] = float(match.group(1))
    
    return scores


def calculate_improvement(
    old_scores: Dict[str, float],
    new_scores: Dict[str, float]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate improvement metrics between two score sets.
    
    Args:
        old_scores: Original scores dictionary
        new_scores: Updated scores dictionary
        
    Returns:
        Dictionary containing improvement deltas and percentage changes
    """
    improvements = {}
    
    for dimension in old_scores:
        if dimension in new_scores:
            old_val = old_scores[dimension]
            new_val = new_scores[dimension]
            delta = new_val - old_val
            percentage = (delta / old_val * 100) if old_val > 0 else 0
            
            improvements[dimension] = {
                "old": old_val,
                "new": new_val,
                "delta": delta,
                "percentage": percentage,
                "improved": delta > 0
            }
    
    return improvements