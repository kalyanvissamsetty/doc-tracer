import pytest

from redline.config import Settings


@pytest.fixture
def settings():
    return Settings(
        azure_openai_endpoint="https://openai.example.test",
        azure_openai_api_key="placeholder",
        azure_openai_deployment_agent="test-agent",
        azure_openai_deployment_vision="test-vision",
        azure_docintel_endpoint="https://docintel.example.test",
        azure_docintel_key="placeholder",
    )
