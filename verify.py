"""④ 검증: 답변에 근거 밖 내용이 섞였는지, 계산이 맞는지 기계적으로 검사한다. LLM을 쓰지 않는다."""
from __future__ import annotations

import ast
import operator
import re

from context import CATEGORIES, Section, load_sections, terms_in

# 인용 괄호: [CF-01] 또는 [CF-01, CF-03]처럼 여러 섹션을 묶은 경우도 인용으로 본다
ID_RE = re.compile(r"[A-Z]{2}-\d{2}")
CITE_RE = re.compile(r"\[\s*[A-Z]{2}-\d{2}(?:\s*[,·/]\s*[A-Z]{2}-\d{2})*\s*\]")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
ADVICE_PATTERNS = [
    r"매수(하세요|하시길|를 추천|해도 좋)", r"매도(하세요|하시길|를 추천)", r"사(세요|셔도 됩니다|도 됩니다)",
    r"팔(으세요|아도 됩니다)", r"(오를|내릴|상승할|하락할) (것입니다|것이다|전망)", r"목표\s*주가는",
]
# 계산식에 쓰일 수 있는 문자(숫자, 소수점, 연산자, 괄호, 공백). 천 단위 쉼표는 미리 지운다.
EXPR_CHARS = r"[\d.()\s+\-−×x*/÷^%]"
CALC_TOLERANCE = 0.005  # 반올림 허용 0.5%


def _strip_thousands(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


def _korean_units(text: str) -> str:
    """'120만원' '2천만' '1.5억' 같은 한국어 단위를 숫자로 바꾼다. 질문은 '2000만원', 답변은 '20,000,000원'으로 쓰는 경우가 많다."""
    units = {"억": 10**8, "천만": 10**7, "백만": 10**6, "만": 10**4}

    def conv(m):
        v = float(m.group(1)) * units[m.group(2)]
        return str(int(v)) if v.is_integer() else str(v)
    return re.sub(r"(\d+(?:\.\d+)?)\s*(억|천만|백만|만)", conv, _strip_thousands(text))


def _numbers(text: str) -> set[str]:
    text = _korean_units(ID_RE.sub(" ", CITE_RE.sub(" ", text)))
    text = re.sub(r"(?m)^\s*\d+\.\s", " ", text)                     # 목록 번호
    text = re.sub(r"K-IFRS 제\d{4}호|제\d+조|문단 [\d.A~, ]+", " ", text)  # 기준서·조문 번호
    text = re.sub(r"[A-Za-z_]+\d+", " ", text)                        # C1, CF_t 같은 기호 첨자
    text = _strip_thousands(text)
    nums = {n.rstrip("0").rstrip(".") if "." in n else n for n in NUM_RE.findall(text)}
    # 백분율은 소수로도 쓴다: 질문의 '5%'를 답변이 공식에 넣을 때 '0.05'로 쓰는 경우
    nums |= {f"{float(p) / 100:g}" for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)}
    return nums


# ---------------------------------------------------------------------------
# 계산 검증: 'A × B = C' 꼴을 찾아 다시 계산한다
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(expr: str) -> float | None:
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**").replace("x", "*")
    expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr).strip()
    if not expr or not re.search(r"\d", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        return float(ev(tree))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def check_calculations(answer: str) -> tuple[list[str], set[str]]:
    """계산식을 다시 계산해 틀린 식 목록과, 검증된 계산 결과 숫자 집합을 돌려준다."""
    errors, derived = [], set()
    for line in _strip_thousands(ID_RE.sub(" ", CITE_RE.sub(" ", answer))).splitlines():
        segs = re.split(r"=|≈", line)
        for left_seg, right_seg in zip(segs, segs[1:]):
            lm = re.search(rf"{EXPR_CHARS}+$", left_seg)
            rm = re.match(rf"\s*{EXPR_CHARS}+", right_seg)
            if not lm or not rm:
                continue
            left, right = lm.group(0).strip(), rm.group(0).strip()
            if not re.search(r"\d\s*[+\-−×x*/÷^]\s*\(?\d", left):  # 연산이 없는 식(예: 'NPV = 41.32')은 건너뛴다
                continue
            lv, rv = _eval(left), _eval(right)
            if lv is None or rv is None:
                continue
            if abs(lv - rv) > max(CALC_TOLERANCE * abs(lv), 0.01):
                errors.append(f"{left} = {right} (다시 계산하면 {lv:,.2f})")
            else:
                derived |= _numbers(right)
    return errors, derived


def verify(answer: str, sections: list[Section], question: str, category: str | None = None) -> dict:
    evidence = "\n".join(s.body for s in sections)
    allowed_ids = {s.id for s in sections}
    checks: list[dict] = []

    # 1. 출처 표시: 문단(빈 줄로 구분)·목록 항목마다 근거 ID가 있어야 한다
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n(?=\s*[-*•]|\s*\d+\.)", answer) if b.strip()]
    uncited = [b for b in blocks if not CITE_RE.search(b) and len(b) > 15]
    checks.append({
        "name": "출처 표시",
        "passed": not uncited and bool(CITE_RE.search(answer)),
        "detail": "모든 문단에 근거 표기 있음" if not uncited else f"근거 표기 없는 문단 {len(uncited)}개: " + " / ".join(u[:40] for u in uncited),
    })

    # 2. 인용 범위: 인용한 섹션이 이번에 조립한 근거 안에 있는가
    cited = set(cited_ids(answer))
    outside = sorted(cited - allowed_ids)
    checks.append({
        "name": "인용 범위",
        "passed": not outside,
        "detail": f"인용 {sorted(cited)}" if not outside else f"조립하지 않은 섹션 인용: {outside}",
    })

    # 3. 카테고리 일치: 인용한 섹션이 판정된 카테고리의 매핑 섹션인가
    if category and category in CATEGORIES:
        foreign = sorted(cited - set(CATEGORIES[category]["sections"]))
        checks.append({
            "name": "카테고리 일치",
            "passed": not foreign,
            "detail": f"인용 섹션이 모두 {category} 매핑 안에 있음" if not foreign else f"{category} 밖 섹션 인용: {foreign}",
        })

    # 4. 계산 검증: 계산식을 다시 계산해 결과가 맞는가
    calc_errors, derived = check_calculations(answer)
    checks.append({
        "name": "계산 검증",
        "passed": not calc_errors,
        "detail": "계산식이 모두 맞음(또는 계산식 없음)" if not calc_errors else "틀린 계산: " + "; ".join(calc_errors),
    })

    # 5. 수치 근거: 답변의 숫자가 근거·질문에 있거나, 검증을 통과한 계산의 결과인가
    unknown = sorted(_numbers(answer) - _numbers(evidence) - _numbers(question) - derived)
    checks.append({
        "name": "수치 근거",
        "passed": not unknown,
        "detail": "모든 수치가 근거·질문·검증된 계산에 있음" if not unknown else f"근거에 없는 수치: {unknown}",
    })

    # 6. 용어 근거: 답변의 금융 용어가 근거나 질문에 있는가
    ungrounded = sorted(t for t in terms_in(answer) if not re.search(re.escape(t), evidence + question, re.I))
    checks.append({
        "name": "용어 근거",
        "passed": not ungrounded,
        "detail": "답변의 금융 용어가 모두 근거에 있음" if not ungrounded else f"근거에 없는 용어: {ungrounded}",
    })

    # 7. 투자 권유·전망 표현
    advice = [m.group(0) for p in ADVICE_PATTERNS for m in [re.search(p, answer)] if m]
    checks.append({
        "name": "투자 권유 금지",
        "passed": not advice,
        "detail": "권유·전망 표현 없음" if not advice else f"금지 표현: {advice}",
    })

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def cited_ids(answer: str) -> list[str]:
    return [sid for group in CITE_RE.findall(answer) for sid in ID_RE.findall(group)]


def citations(answer: str) -> list[dict]:
    """답변이 인용한 섹션의 출처 목록. 답변(생성)과 근거(검색)를 구분해 보여 주기 위해 코드가 만든다."""
    secs = load_sections()
    seen = []
    for sid in cited_ids(answer):
        if sid in secs and sid not in seen:
            seen.append(sid)
    return [secs[s].citation() for s in seen]


def feedback(result: dict) -> list[str]:
    return [f"{c['name']}: {c['detail']}" for c in result["checks"] if not c["passed"]]


def split_sentences(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    return [p.strip(" -*•") for p in parts if len(CITE_RE.sub("", p).strip(" -*•")) > 5]


def jev_support_check(answer: str, sections: list[Section], jev, threshold: float) -> dict:
    """Jev가 답변 문장마다 '근거가 뒷받침하는가' 확률을 낸다. 하나라도 임계값 미만이면 실패."""
    from prompts import jev_support_questions

    sents = split_sentences(answer)
    if not sents:
        return {"name": "근거 지지(Jev)", "passed": False, "detail": "검사할 문장이 없음"}
    ans = jev.decide({"evidence": {s.id: s.body for s in sections}}, jev_support_questions(sents))
    probs = [ans[f"s{i}"]["noul"] for i in range(len(sents))]
    weak = [(s, p) for s, p in zip(sents, probs) if p < threshold]
    detail = (f"문장 {len(sents)}개 모두 근거 지지 확률 ≥ {threshold} (최저 {min(probs):.2f})" if not weak else
              "근거 지지 확률이 낮은 문장: " + " / ".join(f"\"{CITE_RE.sub('', s).strip()[:50]}\" {p:.2f}" for s, p in weak))
    return {"name": "근거 지지(Jev)", "passed": not weak, "detail": detail, "probs": [round(p, 3) for p in probs]}
