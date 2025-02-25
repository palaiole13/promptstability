import os
import pytest
from promptstability.core import get_api_key

@pytest.mark.describe("Test API Key - OpenAI")
def test_api_key_openai():
    """
    Test that get_api_key() retrieves the correct OpenAI API key from the environment.

    By default, get_api_key() uses 'openai', so it should return the value of OPENAI_API_KEY.
    """
    expected_key = os.getenv("OPENAI_API_KEY")
    assert get_api_key() == expected_key, (
        "The API key fetched does not match the expected value for OpenAI. "
        "Please set the OPENAI_API_KEY environment variable."
    )

@pytest.mark.describe("Test API Key - Unsupported API")
def test_api_key_unsupported():
    """
    Test that get_api_key() raises a ValueError for an unsupported API.

    Passing an API name not in the mapping should result in a ValueError.
    """
    with pytest.raises(ValueError):
        get_api_key("unsupported_api")
