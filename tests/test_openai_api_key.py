import os
import pytest
from promptstability.core import get_openai_api_key

@pytest.mark.describe('Test API Key')
def test_openai_api():
    """Test that the API key function correctly retrieves the API key from environment variables."""
    expected_key = os.getenv("OPENAI_API_KEY")
    assert get_openai_api_key() == expected_key, "The API key fetched does not match the expected value. Please set the OPENAI_API_KEY environment variable."
