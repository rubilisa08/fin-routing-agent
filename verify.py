"""④ 검증: 답변에 근거 밖 내용이 섞였는지 기계적으로 검사한다. LLM을 쓰지 않는다."""
from __future__ import annotations

import re

from context import Section, terms_in

CITE_RE = re.compile(r"\[([A-Z]{2}-\d{2})\]")
# 수치: 1,102,500 / 0.05 / 45 / 200% 등. 섹션 ID([RA-05])와 목록 번호(1.)는 미리 지운다.
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
ADVICE_PATTERNS = [
    r"매수(하세요|하시길|를 추천|해도 좋)", r"매도(하세요|하시길|를 추천)", r"사(세요|셔도 됩니다|도 됩니다)",
    r"팔(으세요|아도 됩니다)", r"(오를|내릴|상승할|하락할) (것입니다|것이다|전망)", r"목표\s*주가는",
]


def _normalize_num(s: str) -> str:
    return s.replace(",", "")


def _numbers(text: str) -> set[str]:
    text = CITE_RE.sub(" ", text)
    text = re.sub(r"(?m)^\s*\d+\.\s", " ", text)          # 목록 번호
    text = re.sub(r"K-IFRS 제\d{4}호", " ", text)           # 기준서 번호
    text = re.sub(r"[A-Za-z]+\d+", " ", text)              # FCF1, C1 같은 기호 첨자
    return {_normalize_num(n) for n in NUM_RE.findall(text)}


def verify(answer: str, sections: list[Section], question: str) -> dict:
    evidence = "\n".join(s.body for s in sections)
    allowed_ids = {s.id for s in sections}
    checks: list[dict] = []
    warnings: list[str] = []

    # 1. 근거 표기: 문단(빈 줄로 구분)·목록 항목마다 근거 ID가 있어야 한다
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n(?=\s*[-*•]|\s*\d+\.)", answer) if b.strip()]
    uncited = [b for b in blocks if not CITE_RE.search(b) and len(b) > 15]
    checks.append({
        "name": "근거 표기",
        "passed": not uncited and bool(CITE_RE.search(answer)),
        "detail": "모든 문단에 근거 표기 있음" if not uncited else f"근거 표기 없는 문단 {len(uncited)}개: " + " / ".join(u[:40] for u in uncited),
    })

    # 2. 인용한 섹션이 이번에 조립한 근거 안에 있는가
    cited = set(CITE_RE.findall(answer))
    outside = sorted(cited - allowed_ids)
    checks.append({
        "name": "인용 범위",
        "passed": not outside,
        "detail": f"인용 {sorted(cited)}" if not outside else f"조립하지 않은 섹션 인용: {outside}",
    })

    # 3. 수치: 답변의 숫자가 근거나 질문에 있는가 (계산 결과는 질문에 숫자가 있을 때만 경고로 완화)
    ev_nums, q_nums = _numbers(evidence), _numbers(question)
    unknown = sorted(n for n in _numbers(answer) - ev_nums - q_nums)
    if unknown and q_nums:
        warnings.append(f"근거·질문에 없는 수치(계산 결과일 수 있음): {unknown}")
        unknown = []
    checks.append({
        "name": "수치 근거",
        "passed": not unknown,
        "detail": "모든 수치가 근거 또는 질문에 있음" if not unknown else f"근거에 없는 수치: {unknown}",
    })

    # 4. 금융 용어: 답변에 나온 용어가 근거나 질문에 있는가
    ungrounded = sorted(t for t in terms_in(answer) if not re.search(re.escape(t), evidence + question, re.I))
    checks.append({
        "name": "용어 근거",
        "passed": not ungrounded,
        "detail": "답변의 금융 용어가 모두 근거에 있음" if not ungrounded else f"근거에 없는 용어: {ungrounded}",
    })

    # 5. 투자 권유·전망 표현
    advice = [m.group(0) for p in ADVICE_PATTERNS for m in [re.search(p, answer)] if m]
    checks.append({
        "name": "투자 권유 금지",
        "passed": not advice,
        "detail": "권유·전망 표현 없음" if not advice else f"금지 표현: {advice}",
    })

    return {"passed": all(c["passed"] for c in checks), "checks": checks, "warnings": warnings}


def feedback(result: dict) -> list[str]:
    return [f"{c['name']}: {c['detail']}" for c in result["checks"] if not c["passed"]]
