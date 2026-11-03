"""Jev(TypeSafe AI) 의사결정 모델 클라이언트.

Jev는 글을 쓰지 않고, 상태(state)와 질문 목록을 받아 질문마다 보정된 확률을 돌려준다.
- noul  : 예/아니오 → '예'일 확률
- choice: 보기 중 하나 → 보기별 확률(합 1)과 선택
이 프로젝트에서는 ① 판정(choice), ② 근거 충분성(noul), ④ 문장별 근거 지지 검사(noul)에 쓰고,
③ 답변 작성은 LLM이 한다.

키와 주소는 .env에서 읽는다.
    JEV_API_KEY=...            # TypeSafe 또는 AIML API에서 받은 키
    JEV_PROVIDER=typesafe      # typesafe(직접) | openrouter | aimlapi | laya. 비우면 키를 보고 고른다

OpenRouter로 Jev를 쓸 때(TypeSafe 계정·대기 없이 가능):
    OPENROUTER_API_KEY=...     # openrouter.ai → Keys
    JEV_MODEL=...              # 비우면 제공처별 기본 모델

Laya(Jev 호환 오픈 가중치 모델, 직접 배포한 서버)를 쓸 때:
    LAYA_API_KEY=...           # Railway Laya 서비스의 Variables 탭
    LAYA_BASE_URL=https://<도메인>.up.railway.app
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import requests
from dotenv import load_dotenv

from llm import CACHE_DIR

load_dotenv()

ENDPOINTS = {
    # 제공처: (결정 API 주소, 기본 모델 이름)
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-latest"),
    "aimlapi": ("https://api.aimlapi.com/v1/decisions", "typesafe/jev"),
    # OpenRouter의 System One 호환 주소. 측정 재현을 위해 버전을 고정한다.
    "openrouter": ("https://openrouter.ai/api/v1/systemone", "typesafe/jev-1.14"),
    # Laya는 서버 주소가 사람마다 다르다(LAYA_BASE_URL). 한국어 질문이라 다국어 체크포인트를 고정한다.
    "laya": ("{base}/v1/systemone", "laya-multilingual"),
}


class JevError(RuntimeError):
    pass


class Jev:
    def __init__(self, provider: str | None = None):
        if os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY"):
            default = "typesafe"
        elif os.getenv("OPENROUTER_API_KEY"):
            default = "openrouter"
        elif os.getenv("LAYA_API_KEY"):
            default = "laya"
        else:
            default = "typesafe"
        self.provider = (provider or os.getenv("JEV_PROVIDER") or default).strip().lower()
        if self.provider not in ENDPOINTS:
            raise ValueError(f"알 수 없는 JEV_PROVIDER: {self.provider}")
        if self.provider == "laya":
            self.key = os.getenv("LAYA_API_KEY")
            base = (os.getenv("LAYA_BASE_URL") or "").rstrip("/")
            self.url = ENDPOINTS["laya"][0].format(base=base) if base else ""
        elif self.provider == "openrouter":
            self.key = os.getenv("OPENROUTER_API_KEY") or os.getenv("JEV_API_KEY")
            self.url = os.getenv("JEV_BASE_URL") or ENDPOINTS["openrouter"][0]
        else:
            self.key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY") or os.getenv("AIMLAPI_KEY")
            self.url = os.getenv("JEV_BASE_URL") or ENDPOINTS[self.provider][0]
        self.model = os.getenv("JEV_MODEL") or ENDPOINTS[self.provider][1]
        self.api_calls = 0
        self.cache_hits = 0
        self.last_model = None  # 응답에 적힌 실제 모델 버전

    @property
    def available(self) -> bool:
        return bool(self.key and self.url)

    def models(self) -> list[str]:
        """키와 주소가 맞는지 확인하는 가벼운 호출(GET /v1/models). OpenRouter는 키 정보 조회로 확인한다."""
        if self.provider == "openrouter":
            resp = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {self.key}"}, timeout=60)
            if resp.status_code >= 400:
                raise JevError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
            return [self.model]
        resp = requests.get(self.url.rsplit("/", 1)[0] + "/models",
                            headers={"Authorization": f"Bearer {self.key}"}, timeout=60)
        if resp.status_code >= 400:
            raise JevError(f"Jev {self.provider} HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        items = data.get("data", data.get("models", data)) if isinstance(data, dict) else data
        return [m.get("id", m.get("name")) if isinstance(m, dict) else m for m in items]

    def decide(self, state, questions: dict) -> dict:
        """질문마다 결과를 돌려준다: {키: {"noul": p} | {"choice": k, "confidence": c, "probabilities": {...}}}"""
        if not self.available:
            raise JevError(".env에 Jev 키와 주소가 없습니다(JEV_API_KEY, 또는 LAYA_API_KEY + LAYA_BASE_URL).")
        body = {"model": self.model, "state": state, "questions": questions}
        use_cache = os.getenv("LLM_CACHE", "1") != "0"
        key = hashlib.sha256(json.dumps([self.provider, body], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = CACHE_DIR / f"jev_{key}.json"
        if use_cache and path.exists():
            self.cache_hits += 1
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = self._post(body)
            self.api_calls += 1
            if use_cache:
                CACHE_DIR.mkdir(exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.last_model = data.get("model", self.model)
        answers = data.get("answers") or {}
        missing = [q for q in questions if q not in answers]
        if missing:
            raise JevError(f"Jev 응답에 없는 질문: {missing}")
        return answers

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(4):
            resp = requests.post(self.url, headers=headers, json=body, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 3)
                continue
            if resp.status_code >= 400:
                # 키가 메시지에 섞여 나오지 않도록 응답 본문만 짧게 남긴다
                raise JevError(f"Jev {self.provider} HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise JevError("Jev 재시도 실패")


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q = {"type": "noul", "instructions": instructions}
    if true and false:
        q["criteria"] = {"true": true, "false": false}
    return q


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}
