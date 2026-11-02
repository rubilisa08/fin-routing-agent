"""문서 쪼개기 · 카테고리별 근거 조립 (파이프라인 ②).

보류(abstain) 판단은 이 단계에서 한다. ① 판정은 카테고리와 신뢰도만 내고,
고른 카테고리의 근거로 답할 수 있는지는 여기서 따로 판단한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).parent / "docs"
ABSTAIN = "abstain"   # 근거 부족·범위 밖 → 답변 보류
CLARIFY = "clarify"   # 판정 신뢰도 낮음 → 질문 되묻기

# ---------------------------------------------------------------------------
# 카테고리 ↔ 근거 문서 매핑표. REPORT.md의 매핑표와 같은 내용이다.
# ---------------------------------------------------------------------------
CATEGORIES: dict[str, dict] = {
    "C1": {
        "focus": "분개·결산 등 거래를 장부에 기록하는 방법",
        "name": "회계원리",
        "en": "Accounting Principles",
        "tool": "lookup_principles",
        "description": "회계등식, 거래의 이중성과 8요소, 차변·대변 기록 규칙, 분개, 회계순환과정과 시산표, "
        "발생주의·현금주의, 결산정리분개(선급·선수·미지급·미수, 감가상각), 마감",
        "sections": [f"AP-{i:02d}" for i in range(1, 9)] + ["FS-03"],
        "out_of_scope": "세무회계(법인세 신고·세무조정), 원가·관리회계, 특정 업종 회계처리, 개별 기준서 세부 처리",
        "scope_patterns": [r"세무\s*조정|세법|세무|법인세\s*신고", r"원가\s*(계산|회계|배부)|제조원가|관리회계",
                           r"리스(?!크)", r"이연법인세", r"수익인식|수행의무"],
        "handoff_to": "세무는 세무사, 원가·관리회계는 해당 교재, 개별 기준서는 K-IFRS 원문",
    },
    "C2": {
        "focus": "재무제표·공시·감사보고서를 읽는 방법",
        "name": "재무제표",
        "en": "Financial Statements",
        "tool": "lookup_statements",
        "description": "재무제표의 구성과 구조(재무상태표·포괄손익계산서·현금흐름표·자본변동표·주석), 유동/비유동 구분, "
        "기타포괄손익, 연결/별도재무제표, 재무제표 간 관계, 배당의 회계처리(비용이 아닌 자본변동표의 이익잉여금 감소), 공시(DART·사업보고서·주요사항보고서)와 감사보고서·감사의견 읽기",
        "sections": [f"FS-{i:02d}" for i in range(1, 11)] + [f"DS-{i:02d}" for i in range(1, 10)],
        "out_of_scope": "리스·수익인식·금융상품·법인세 등 개별 기준서 세부, 특정 회사의 실제 공시 내용·실적, "
        "한국거래소 상장규정에 따른 조치(관리종목 지정, 상장폐지 요건 등)",
        "scope_patterns": [r"리스(?!크)", r"이연법인세", r"수익인식|수행의무", r"상장\s*폐지|상폐|관리\s*종목|거래\s*정지"],
        "handoff_to": "K-IFRS 기준서 원문, DART 원문 공시, 한국거래소 상장규정",
    },
    "C3": {
        "focus": "재무비율로 기업을 분석하는 방법",
        "name": "재무분석",
        "en": "Financial Analysis",
        "tool": "lookup_analysis",
        "description": "재무비율의 정의·계산·해석: 수익성(ROE·ROA·이익률), 듀폰분석, 유동성(유동비율·당좌비율), "
        "안정성(부채비율·이자보상배율·자본잠식), 활동성(회전율·현금전환주기), 성장성, 현금흐름 지표, 재무분석의 한계",
        "sections": [f"RA-{i:02d}" for i in range(1, 10)] + ["FS-02"],
        "out_of_scope": "특정 업종의 평균값, 특정 기업의 실제 비율 수치, 신용등급 기준",
        "scope_patterns": [r"업종\s*평균|업계\s*평균|평균\s*\S*(비율|률)", r"신용\s*등급"],
        "handoff_to": "한국은행 기업경영분석, 업종 리포트, 신용평가사 평가방법론",
    },
    "C4": {
        "focus": "투자결정·자금조달·가치평가",
        "name": "기업재무",
        "en": "Corporate Finance",
        "tool": "lookup_corpfin",
        "description": "화폐의 시간가치(현재가치·미래가치·복리·연금), 자본예산(NPV·IRR·회수기간), 자금조달(부채·자기자본), "
        "자본구조(MM이론·상충이론), 레버리지(DOL·DFL), 자본비용(WACC·CAPM), DCF·멀티플(PER·PBR·EV/EBITDA), EPS, 배당 지표(배당성향·배당수익률·배당할인모형), 채권",
        "sections": [f"VA-{i:02d}" for i in range(1, 11)] + [f"CF-{i:02d}" for i in range(1, 7)] + ["FS-04"],
        "out_of_scope": "특정 시점의 시장 수치(무위험이자율, 시장위험프리미엄, 시장금리, 업종 평균 멀티플), "
        "특정 기업의 실제 투자안 평가, 특정 종목의 목표주가·매수·매도 판단",
        "scope_patterns": [r"(요즘|최근|현재|지금|올해)[^?？]{0,30}(몇\s*%|얼마|수치)", r"목표\s*주가"],
        "handoff_to": "평가 기준일의 시장 자료(한국공인회계사회, 한국은행 등) 또는 전문가",
    },
    "C0": {
        "focus": "지원하지 않는 질문",
        "name": "지원하지 않는 질문",
        "en": "Unsupported",
        "tool": None,
        "description": "위 네 분야 어디에도 속하지 않는 질문: 개별 종목 매수·매도 추천, 주가·시장 전망, 개인 세금 신고·연말정산, "
        "특정 회사의 실적 수치 조회, 금융·회계와 무관한 질문",
        "sections": [],
        "out_of_scope": "학습노트 전체",
        "scope_patterns": [],
        "handoff_to": "세무 상담은 세무사, 투자 판단은 본인 또는 투자자문업자, 실적 수치는 DART 원문",
    },
}
TOOL_TO_CATEGORY = {c["tool"]: k for k, c in CATEGORIES.items() if c["tool"]}

# 어느 카테고리든 학습노트가 다루지 않는 요청 (각 문서 '범위' 문단의 공통분모)
GLOBAL_SCOPE_PATTERNS = [
    r"(사도|살까|팔까|사야 할까|팔아야 할까|매수|매도)\s*(돼|될까|되나|할까|해도|하나)?",
    r"(19|20)\d{2}\s*년\s*(도)?\s*\S*\s*(실적|영업이익|매출|순이익|주가)",
    r"(주가|코스피|코스닥|환율)\s*(가|는)?\s*(오를|내릴|전망|예측)",
    r"(우리|저희)\s*회사",
]

# 금융·회계 용어 사전. 질문이나 답변에 나온 용어가 근거에 없으면 '근거 밖'으로 본다.
FIN_TERMS = [
    r"리스(?!크)", "사용권자산", "리스부채", "이연법인세", "영업권", "충당부채", "우발부채", "파생상품",
    "공정가치위계", "수익인식", "수행의무", "대손충당금", "기대신용손실", "정률법", "생산량비례법",
    "선입선출", "후입선출", "총평균법", "손상차손", "재평가잉여금", "재측정요소", "비지배지분", "지분법", "원가법",
    "세무조정", "손금", "익금", "종합소득세", "금융소득", "종합과세", "분리과세", "양도소득세", "증권거래세", "부가가치세",
    "제조간접비", "표준원가", "공헌이익", "손익분기점",
    "관리종목", "상장폐지", "목표주가", "시장위험프리미엄", "무위험이자율", "공매도", "신용등급",
    "핵심감사사항", "계속기업", "의견거절", "부적정", "한정의견", "강조사항", "조회공시", "공정공시",
    "주요사항보고서", "사업보고서", "분기보고서", "반기보고서", "정정공시",
    "분개", "시산표", "결산정리", "수정분개", "선급비용", "선수수익", "미지급비용", "미수수익", "감가상각누계액",
    "대차평균", "총계정원장", "마감분개", "거래의 8요소",
    "현금전환주기", "듀폰", "이자보상배율", "당좌비율", "유동비율", "부채비율", "자본잠식",
    "잉여현금흐름", "재무레버리지", "총자산회전율", "재고자산회전율", "매출채권회전율",
    "NPV", "IRR", "MIRR", "수익성지수", "회수기간", "자본예산", "MM", "상충이론", "자본조달순서", "대리인비용",
    "영업레버리지", "결합레버리지", "DOL", "DFL", "DCL",
    "WACC", "CAPM", "DCF", "EBITDA", "PER", "PBR", "PSR", "EPS", "BPS", "듀레이션", "볼록성", "만기수익률",
    "잔존가치", "영구성장률", "배당성향", "배당수익률", "희석주당이익", "베타",
    "자사주 소각", "스톡옵션", "전환사채", "신주인수권", "유상증자", "무상증자", "감자",
    "IFRS 18", "K-IFRS 제\\d{4}호",
]


@dataclass
class Section:
    id: str
    title: str
    text: str      # 출처 줄을 뺀 본문
    doc: str       # 파일명
    doc_title: str
    source: str    # 섹션의 출처(기준서 문단·법령 조문·교재)
    version: str

    @property
    def body(self) -> str:
        return f"[{self.id}] {self.title}\n{self.text}"

    def citation(self) -> dict:
        return {"id": self.id, "title": self.title, "doc": self.doc_title, "source": self.source, "version": self.version}


@dataclass
class Settings:
    """실험 손잡이. 개선 시도는 한 번에 하나만 바꾼다."""

    top_k: int = 4                  # 카테고리 안에서 프롬프트에 넣을 섹션 수
    min_similarity: float = 0.05    # offline 게이트: 최고 유사도가 이보다 낮으면 보류
    confidence_threshold: float = 0.75  # ① 판정 신뢰도가 이보다 낮으면 되묻기 (0.6~0.96 비교 후 선택, REPORT §4)
    use_scope_rules: bool = True    # 문서 '범위' 문단에서 뽑은 규칙으로 보류
    use_term_coverage: bool = True  # 질문의 금융 용어가 카테고리 근거에 없으면 보류
    use_llm_gate: bool = True       # (LLM 모드) 근거 충분성 LLM 판단

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Assembly:
    category: str
    tool: str | None
    sections: list[Section] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    sufficient: bool = False
    handoff_reason: str = ""
    gate: list[dict] = field(default_factory=list)  # 보류 판단 근거 기록


# ---------------------------------------------------------------------------
# 문서 쪼개기
# ---------------------------------------------------------------------------
SECTION_RE = re.compile(r"^## \[([A-Z]{2}-\d{2})\]\s*(.+)$", re.M)
SOURCE_RE = re.compile(r"^> 출처:\s*(.+)$", re.M)


@lru_cache(maxsize=1)
def load_sections() -> dict[str, Section]:
    sections: dict[str, Section] = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        doc_title = raw.splitlines()[0].lstrip("# ").strip()
        version = (re.search(r"^> 버전:\s*(.+)$", raw, re.M) or [None, ""])[1]
        heads = list(SECTION_RE.finditer(raw))
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
            chunk = raw[m.end():end]
            src = SOURCE_RE.search(chunk)
            sections[m.group(1)] = Section(
                id=m.group(1), title=m.group(2).strip(), text=SOURCE_RE.sub("", chunk).strip(), doc=path.name,
                doc_title=doc_title, source=src.group(1).strip() if src else "", version=version,
            )
    return sections


def check_mapping() -> list[str]:
    """매핑표 점검: 매핑된 섹션이 실제로 있는지, 어느 카테고리에도 연결되지 않은 섹션, 출처 없는 섹션이 있는지."""
    secs = load_sections()
    problems = []
    mapped = set()
    for key, cat in CATEGORIES.items():
        if key != "C0" and not cat["sections"]:
            problems.append(f"{key}: 연결된 근거 섹션이 없음")
        for sid in cat["sections"]:
            mapped.add(sid)
            if sid not in secs:
                problems.append(f"{key}: 매핑된 {sid}가 문서에 없음")
    for sid, s in secs.items():
        if sid not in mapped:
            problems.append(f"{sid}: 어느 카테고리에도 연결되지 않음")
        if not s.source:
            problems.append(f"{sid}: 출처 줄이 없음")
    return problems


# ---------------------------------------------------------------------------
# 검색 (카테고리 안에서만)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _vectorizer() -> TfidfVectorizer:
    docs = [s.body for s in load_sections().values()]
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True).fit(docs)


def similarity(query: str, texts: list[str]) -> list[float]:
    vec = _vectorizer()
    return cosine_similarity(vec.transform([query]), vec.transform(texts))[0].tolist()


def rank_sections(question: str, section_ids: list[str]) -> list[tuple[Section, float]]:
    secs = load_sections()
    pool = [secs[s] for s in section_ids]
    scores = similarity(question, [s.body for s in pool])
    return sorted(zip(pool, scores), key=lambda x: -x[1])


def terms_in(text: str) -> set[str]:
    found = set()
    for t in FIN_TERMS:
        m = re.search(t, text, re.I)
        if m:
            found.add(m.group(0))
    return found


# ---------------------------------------------------------------------------
# ② 근거 조립 + 보류 판단
# ---------------------------------------------------------------------------
def assemble(question: str, category: str, settings: Settings, llm=None) -> Assembly:
    cat = CATEGORIES[category]
    asm = Assembly(category=category, tool=cat["tool"])

    if not cat["tool"]:
        asm.handoff_reason = "학습노트가 다루지 않는 범위 밖 질문입니다."
        asm.gate.append({"check": "category", "passed": False, "detail": "C0(지원하지 않는 질문)"})
        return asm

    ranked = rank_sections(question, cat["sections"])
    asm.scores = {s.id: round(score, 3) for s, score in ranked}
    asm.sections = [s for s, _ in ranked[: settings.top_k]]

    # (a) 규칙: 각 문서의 '범위' 문단에 적힌, 다루지 않는 요청인가
    if settings.use_scope_rules:
        for pat in GLOBAL_SCOPE_PATTERNS + cat["scope_patterns"]:
            m = re.search(pat, question)
            if m:
                asm.gate.append({"check": "scope_rule", "passed": False, "detail": f"범위 밖 신호 '{m.group(0)}'"})
                asm.handoff_reason = f"학습노트가 다루지 않는 내용입니다 — {cat['out_of_scope']}"
                return asm
        asm.gate.append({"check": "scope_rule", "passed": True, "detail": "범위 밖 신호 없음"})

    # (b) 용어 커버리지: 질문의 금융 용어가 카테고리 근거 어디에도 없으면 답할 수 없다
    if settings.use_term_coverage:
        cat_text = "\n".join(load_sections()[s].body for s in cat["sections"])
        missing = sorted(t for t in terms_in(question) if not re.search(re.escape(t), cat_text, re.I))
        if missing:
            asm.gate.append({"check": "term_coverage", "passed": False, "detail": f"근거에 없는 용어: {', '.join(missing)}"})
            asm.handoff_reason = f"질문의 핵심 용어({', '.join(missing)})를 다루는 근거가 없습니다."
            return asm
        asm.gate.append({"check": "term_coverage", "passed": True, "detail": "질문 용어가 모두 근거에 있음"})

    # (c) 근거 충분성: LLM 모드는 LLM이 판단하고 필요한 섹션만 고른다. offline은 유사도 하한.
    if llm is not None and llm.online and settings.use_llm_gate:
        from prompts import sufficiency_prompt

        system, user = sufficiency_prompt(question, cat, asm.sections)
        verdict = llm.json(system, user)
        used = [s for s in asm.sections if s.id in set(verdict.get("needed_sections", []))]
        ok = bool(verdict.get("sufficient")) and bool(used)
        asm.gate.append({"check": "llm_gate", "passed": ok, "detail": verdict.get("reason", "")})
        if not ok:
            asm.handoff_reason = verdict.get("missing") or verdict.get("reason") or "근거가 부족합니다."
            return asm
        asm.sections = used
    else:
        best = max(asm.scores.values(), default=0.0)
        ok = best >= settings.min_similarity
        asm.gate.append({"check": "similarity", "passed": ok, "detail": f"최고 유사도 {best:.3f} (하한 {settings.min_similarity})"})
        if not ok:
            asm.handoff_reason = "질문과 충분히 관련된 근거를 찾지 못했습니다."
            return asm

    asm.sufficient = True
    return asm
