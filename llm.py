"""LLM 백엔드. Gemini(기본) / OpenAI / offline(키 없이 규칙 기반) 중 하나를 고른다.

LLM_PROVIDER 환경변수로 강제할 수 있고, 없으면 설정된 API 키를 보고 자동 선택한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class QuotaExhausted(RuntimeError):
    """하루 무료 호출 한도 소진. 기다려도 풀리지 않으므로 실행을 멈춘다."""


# 같은 (모델, 프롬프트) 호출은 디스크에 저장해 다시 쓴다. temperature=0이라 결과가 같고,
# 무료 한도(모델당 하루 20회) 안에서 측정을 여러 날에 이어서 할 수 있다. LLM_CACHE=0이면 끈다.
CACHE_DIR = Path(__file__).parent / ".llm_cache"


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
        self.api_calls = 0   # 실제 API 호출 수
        self.cache_hits = 0  # 캐시에서 꺼낸 수
        if self.provider == "gemini":
            from google import genai
            from google.genai import types

            # 최신 3.7·3.8 Flash는 과부하(503)가 잦고, flash-lite는 같은 입력에 분류가 흔들려서 3.5 Flash를 기본으로 쓴다.
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
            self._client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                # SDK 내부 재시도는 끄고(한 호출이 몇 분씩 걸림) 아래 complete()에서 직접 재시도한다
                http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)),
            )
        elif self.provider == "openai":
            from openai import OpenAI

            self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
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
        use_cache = os.getenv("LLM_CACHE", "1") != "0"
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, user, as_json]).encode()).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if use_cache and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self._call_with_retry(system, user, as_json)
        self.api_calls += 1
        if use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps({"model": self.model, "text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def _call_with_retry(self, system: str, user: str, as_json: bool) -> str:
        for attempt in range(4):
            try:
                return self._call(system, user, as_json)
            except Exception as e:  # 429·일시 장애는 잠깐 쉬고 재시도
                if re.search(r"PerDay|per day", str(e)):  # 하루 한도 초과는 기다려도 소용없다
                    raise QuotaExhausted(f"{self.model} 하루 무료 호출 한도를 다 썼습니다. 내일 다시 하거나 결제를 등록하세요.") from e
                if attempt == 3 or not re.search(r"429|503|504|rate|quota|unavailable|deadline|timed? ?out", str(e), re.I):
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
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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
