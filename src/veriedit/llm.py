from __future__ import annotations

import json
import os
from typing import Any

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency/runtime dependent
    genai = None
    genai_types = None


def has_gemini_support() -> bool:
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


class GeminiStructuredClient:
    def __init__(self, model: str) -> None:
        if not has_gemini_support():  # pragma: no cover - dependency/runtime dependent
            raise RuntimeError("Gemini is unavailable. Set GEMINI_API_KEY and install google-genai.")
        self.model = model
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": json.dumps({"system": system_prompt, "payload": payload}, indent=2)}]}
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        text = getattr(response, "text", "") or ""
        return json.loads(text)
