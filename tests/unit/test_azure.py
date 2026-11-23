import json
from unittest.mock import Mock

import httpx
import openai
import pytest

from redline.config import ConfigurationError, Settings, load_settings
from redline.llm.azure_client import AzureChat, ModelRequestError, TokenRates, build_client, image_part
from redline.ocr.docintel import DocumentIntel
from redline.storage import LocalStorage


def test_settings_precedence_and_safe_errors(tmp_path, monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("AZURE_OPENAI_API_KEY=dotenv-test-key\nMAX_AGENT_STEPS=20\n")
    with pytest.raises(ConfigurationError, match="AZURE_DOCINTEL_KEY \\(missing\\)") as failure:
        load_settings(dotenv)
    assert "dotenv-test-key" not in str(failure.value)
    streamlit = {
        "AZURE_OPENAI_ENDPOINT": "https://openai.example.test/",
        "AZURE_OPENAI_DEPLOYMENT_AGENT": "agent",
        "AZURE_OPENAI_DEPLOYMENT_VISION": "vision",
        "AZURE_DOCINTEL_ENDPOINT": "https://docintel.example.test",
        "AZURE_DOCINTEL_KEY": "test-doc-key",
        "MAX_AGENT_STEPS": 30,
    }
    monkeypatch.setenv("MAX_AGENT_STEPS", "40")
    config = load_settings(dotenv, streamlit_secrets=streamlit)
    assert config.max_agent_steps == 40
    assert config.azure_openai_api_version == "v1"
    assert config.azure_openai_endpoint == "https://openai.example.test"
    assert "dotenv-test-key" not in repr(config)
    monkeypatch.setenv("MAX_JOB_COST_USD", "invalid-secret-like-value")
    with pytest.raises(ConfigurationError, match="MAX_JOB_COST_USD \\(invalid value\\)") as failure:
        load_settings(dotenv, streamlit_secrets=streamlit)
    assert "invalid-secret-like-value" not in str(failure.value)


def test_client_routes_to_foundry_v1_or_dated_api(settings):
    with build_client(settings) as current:
        assert str(current.base_url) == "https://openai.example.test/openai/v1/"
        assert current.max_retries == 2
    with build_client(settings.model_copy(update={"azure_openai_api_version": "2024-10-21"})) as dated:
        assert isinstance(dated, openai.AzureOpenAI)
        assert dated.default_query["api-version"] == "2024-10-21"


def _client(handler):
    return openai.OpenAI(
        base_url="https://example.test/openai/v1/", api_key="placeholder",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0,
    )


def _completion(message, finish_reason="stop"):
    return {
        "id": "generated", "object": "chat.completion", "created": 1, "model": "test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def test_tool_call_round_trip_and_usage(settings):
    tool = {"type": "function", "function": {
        "name": "lookup", "description": "Returns an item number",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }}
    call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/openai/v1/chat/completions"
        assert payload["model"] == "test-agent" and payload["tools"] == [tool]
        if len(payload["messages"]) == 1:
            result = _completion({"role": "assistant", "tool_calls": [call]}, "tool_calls")
        else:
            assert payload["messages"][1]["tool_calls"] == [call]
            assert payload["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}
            result = _completion({"role": "assistant", "content": "42"})
        return httpx.Response(200, json=result)

    with _client(handler) as client:
        wrapper = AzureChat(settings, client=client, rates={"test-agent": TokenRates(3, 10)})
        messages = [{"role": "user", "content": "Look up the item number."}]
        result = wrapper.chat(messages, tools=[tool], max_completion_tokens=100)
        assert result.completion.choices[0].message.tool_calls[0].function.name == "lookup"
        assert (result.input_tokens, result.output_tokens) == (20, 10)
        assert result.estimated_usd == pytest.approx(0.00016)
        assistant = result.completion.choices[0].message.model_dump(exclude_none=True)
        answer = wrapper.chat(messages + [assistant, {"role": "tool", "tool_call_id": "call_1", "content": "42"}], tools=[tool])
        assert answer.completion.choices[0].message.content == "42"


def test_image_input_and_safe_error(settings):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "test-vision"
        assert "tools" not in payload
        assert payload["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,UE5H"
        return httpx.Response(200, json=_completion({"role": "assistant", "content": "red"}))

    with _client(handler) as client:
        result = AzureChat(settings, client=client).chat(
            [{"role": "user", "content": [{"type": "text", "text": "Color?"}, image_part(b"PNG")]}],
            role="vision",
        )
        assert result.completion.choices[0].message.content == "red"
        assert result.estimated_usd is None
    with _client(lambda request: httpx.Response(401, json={"error": {"message": "credential-detail"}})) as client:
        with pytest.raises(ModelRequestError, match="HTTP 401") as failure:
            AzureChat(settings, client=client).chat([{"role": "user", "content": "test"}])
        assert "credential-detail" not in str(failure.value)


def test_docintel_caches_by_content_page_and_model(settings, tmp_path):
    storage = LocalStorage(tmp_path / "artifacts")
    client = Mock()

    def analyze(model, *, body, pages, content_type, **options):
        assert body.read() in (b"first PDF", b"revised PDF")
        assert content_type == "application/octet-stream"
        assert bool(options.get("features")) == (model == "prebuilt-layout")
        result = {"modelId": model, "pages": [{"pageNumber": int(pages), "words": []}], "tables": []}
        poller = Mock()
        poller.result.return_value.as_dict.return_value = result
        return poller

    client.begin_analyze_document.side_effect = analyze
    first, duplicate = tmp_path / "first.pdf", tmp_path / "copy.pdf"
    first.write_bytes(b"first PDF")
    duplicate.write_bytes(first.read_bytes())
    ocr = DocumentIntel(settings, storage, client=client)
    reserve = Mock()
    assert ocr.extract(first, 1, before_request=reserve) == ocr.extract(duplicate, 1, before_request=reserve)
    reserve.assert_called_once()
    assert client.begin_analyze_document.call_count == 1
    assert ocr.extract(first, 2)["pages"][0]["pageNumber"] == 2
    assert ocr.extract(first, 1, mode="read")["modelId"] == "prebuilt-read"
    first.write_bytes(b"revised PDF")
    ocr.extract(first, 1)
    assert client.begin_analyze_document.call_count == 4


def test_local_storage_rejects_paths_outside_root(tmp_path):
    storage = LocalStorage(tmp_path / "cache")
    with pytest.raises(ValueError, match="inside"):
        storage.write("../outside", b"data")


def test_region_reader_uses_strict_schema_and_keeps_low_confidence(settings):
    from redline.llm.region_reader import RegionContext, read_region_pair

    response = {"outcome": "modification", "field": "Pressure", "v1_value": "100 psi",
                "v2_value": "120 psi", "confidence": 0.63}

    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "test-vision"
        assert payload["temperature"] == 0
        contract = payload["response_format"]["json_schema"]
        assert contract["strict"] is True
        assert contract["schema"]["additionalProperties"] is False
        assert set(contract["schema"]["required"]) == set(response)
        assert "bbox" not in contract["schema"]["properties"]
        assert "untrusted document data" in payload["messages"][0]["content"]
        crops = [part["image_url"]["url"] for part in payload["messages"][1]["content"]
                 if part["type"] == "image_url"]
        assert crops == ["data:image/png;base64,VjE=", "data:image/png;base64,VjI="]
        return httpx.Response(200, json=_completion({"role": "assistant", "content": json.dumps(response)}))

    with _client(handler) as client:
        result = read_region_pair(
            b"V1", b"V2", RegionContext("Pixel diff region 4", "Pressure"),
            chat=AzureChat(settings, client=client),
        )
    assert result.interpretation.v2_value == "120 psi"
    assert result.interpretation.needs_review
    assert result.evidence == "Pixel diff region 4"
    assert result.request.input_tokens == 20


@pytest.mark.parametrize(("content", "finish_reason"), [
    ('{"field": "unexpected-format"}', "stop"),
    ('{"outcome":"uncertain","field":null,"v1_value":null,"v2_value":null,"confidence":1.7}', "stop"),
    ('{"outcome":"unchanged","field":null,"v1_value":null,"v2_value":null,"confidence":1,"bbox":[0,0,1,1]}', "stop"),
    (None, "content_filter"),
])
def test_region_reader_rejects_invalid_output(settings, content, finish_reason):
    from redline.llm.region_reader import RegionContext, read_region_pair

    message = {"role": "assistant", "content": content}
    with _client(lambda req: httpx.Response(200, json=_completion(message, finish_reason))) as client:
        wrapper = AzureChat(settings, client=client)
        with pytest.raises(ValueError, match="deterministic evidence"):
            read_region_pair(b"V1", b"V2", RegionContext(" "), chat=wrapper)
        with pytest.raises(ModelRequestError, match="region reading"):
            read_region_pair(b"V1", b"V2", RegionContext("Changed pixels"), chat=wrapper)
