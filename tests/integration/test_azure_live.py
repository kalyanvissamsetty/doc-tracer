"""Opt-in checks, using generated content and consuming Azure quota."""
from io import BytesIO
import json
import os

import pymupdf
from PIL import Image
import pytest

from redline.config import ConfigurationError, load_settings
from redline.llm.azure_client import AzureChat, build_client, image_part
from redline.ocr.docintel import DocumentIntel
from redline.storage import LocalStorage

pytestmark = pytest.mark.skipif(
    os.getenv("REDLINE_LIVE_AZURE") != "1", reason="Set REDLINE_LIVE_AZURE=1 to make live Azure requests.",
)


@pytest.fixture(scope="module")
def live_settings():
    try:
        return load_settings()
    except ConfigurationError as exc:
        pytest.fail(str(exc), pytrace=False)


def test_live_azure_tool_call(live_settings):
    tools = [{"type": "function", "function": {
        "name": "read_item_code", "description": "Retrieve the current item code.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }}]
    messages = [{"role": "user", "content": (
        "Call read_item_code to retrieve the item code. After the tool responds, repeat only the code."
    )}]
    with build_client(live_settings) as sdk:
        chat = AzureChat(live_settings, client=sdk)
        reply = chat.chat(messages, tools=tools, max_completion_tokens=512)
        message = reply.completion.choices[0].message
        assert message.tool_calls and len(message.tool_calls) == 1
        call = message.tool_calls[0]
        assert call.type == "function" and call.function.name == "read_item_code"
        assert json.loads(call.function.arguments) == {}
        followup = messages + [
            message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "ITEM-4791"},
        ]
        answer = chat.chat(followup, max_completion_tokens=512)
        assert "ITEM-4791" in (answer.completion.choices[0].message.content or "")


def test_live_azure_image_input(live_settings):
    buffer = BytesIO()
    Image.new("RGB", (128, 128), "red").save(buffer, format="PNG")
    with build_client(live_settings) as sdk:
        reply = AzureChat(live_settings, client=sdk).chat(
            [{"role": "user", "content": [
                {"type": "text", "text": "What is the dominant color of this image? Answer in one word."},
                image_part(buffer.getvalue()),
            ]}],
            role="vision", max_completion_tokens=512,
        )
    assert "red" in (reply.completion.choices[0].message.content or "").casefold()


def test_live_docintel_layout_and_read(live_settings, tmp_path):
    path = tmp_path / "generated.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Inspection record")
        pdf.save(path)
    ocr = DocumentIntel(live_settings, LocalStorage(tmp_path / "cache"))
    try:
        for mode in ("layout", "read"):
            result = ocr.extract(path, 1, mode=mode)
            assert "Inspection record" in result["content"]
            assert ocr.extract(path, 1, mode=mode) == result
    finally:
        ocr.client.close()


def test_live_azure_region_reading(live_settings):
    from PIL import ImageDraw, ImageFont

    from redline.llm.region_reader import RegionContext, read_region_pair

    def crop(value):
        image = Image.new("RGB", (450, 110), "white")
        ImageDraw.Draw(image).text((10, 20), f"Pressure: {value} psi", fill="black", font=ImageFont.load_default(32))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    with build_client(live_settings) as sdk:
        result = read_region_pair(
            crop(100), crop(120), RegionContext("The number in the pressure field changed.", "Pressure"),
            chat=AzureChat(live_settings, client=sdk), temperature=None,
        )
    assert result.interpretation.outcome == "modification"
    assert "100" in (result.interpretation.v1_value or "")
    assert "120" in (result.interpretation.v2_value or "")
