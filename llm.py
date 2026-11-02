"""LLM 백엔드. Gemini(기본) / OpenAI / offline(키 없이 규칙 기반) 중 하나를 고른다.

LLM_PROVIDER 환경변수로 강제할 수 있고, 없으면 설정된 API 키를 보고 자동 선택한다.
"""
from __future__ import annotations

import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()


def resolve_provider() -> str:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced:
        return forced
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "offline"


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


class LLM:
    def __init__(self, provider: str | None = None):
        self.provider = provider or resolve_provider()
        if self.provider == "gemini":
            from google import genai

            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self._client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
        elif self.provider == "openai":
            from openai import OpenAI

            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self._client = OpenAI()
        elif self.provider == "offline":
            self.model = "rule-based"
        else:
            raise ValueError(f"알 수 없는 LLM_PROVIDER: {self.provider}")

    @property
    def online(self) -> bool:
        return self.provider != "offline"

    def complete(self, system: str, user: str, as_json: bool = False) -> str:
        if not self.online:
            raise RuntimeError("offline 모드에서는 LLM을 호출할 수 없습니다.")
        for attempt in range(4):
            try:
                return self._call(system, user, as_json)
            except Exception as e:  # 429·일시 장애는 잠깐 쉬고 재시도
                if attempt == 3 or not re.search(r"429|503|rate|quota|unavailable", str(e), re.I):
                    raise
                time.sleep(2 ** attempt * 5)
        raise RuntimeError("unreachable")

    def json(self, system: str, user: str) -> dict:
        return _parse_json(self.complete(system, user, as_json=True))

    def _call(self, system: str, user: str, as_json: bool) -> str:
        if self.provider == "gemini":
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json" if as_json else "text/plain",
            )
            resp = self._client.models.generate_content(
                model=self.model, contents=user, config=config
            )
            return resp.text or ""
        kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""
