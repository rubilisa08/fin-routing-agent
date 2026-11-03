"""FinRoute AI 파이프라인.

① 카테고리 판정 → (신뢰도 점검: 낮으면 되묻기) → ② 근거 조립(근거 부족이면 보류) → ③ 근거만으로 답변 → ④ 검증

LangGraph StateGraph로 잇는다. 검증 실패 시 한 번 다시 답하고, 그래도 실패하면 보류한다.
카테고리를 고르는 일(classify)과 되묻기·보류 판단(check_confidence, gather)은 서로 다른 노드다.
"""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from context import ABSTAIN, CATEGORIES, CLARIFY, Section, Settings, assemble, load_sections, similarity
from llm import LLM
from prompts import EVIDENCE_TYPES, OFFLINE_KEYWORDS, OFFLINE_OTHER_SIGNALS, answer_prompt, jev_router_questions, router_prompt
from verify import citations, feedback, jev_support_check, verify

MAX_ANSWER_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    question: str
    category: str                # ① 판정 결과
    confidence: float            # ① 판정 신뢰도(0~1)
    alternative: str | None      # ① 두 번째 후보 카테고리
    evidence_types: list[str]    # ① 필요한 근거 유형
    category_reason: str
    probabilities: dict | None   # (Jev) 카테고리별 확률
    tool: str | None             # ② 조회한 근거 도구
    sections: list[Section]      # ② 프롬프트에 넣은 근거 섹션
    candidates: list[str]        # ② 검색 상위 top_k 후보 섹션 ID
    scores: dict[str, float]     # ② 카테고리 안 섹션별 유사도
    gate: list[dict]             # ② 보류 판단 기록
    sufficient: bool
    handoff_reason: str
    answer: str                  # ③ 생성된 답변
    attempts: int
    failed_drafts: list[dict]    # ④에서 걸러진 답변 초안과 실패 사유
    verification: dict           # ④ 마지막 검증 결과
    citations: list[dict]        # 답변이 인용한 근거의 출처(코드가 만든다)
    status: str                  # "answered" | "abstained" | "clarify"
    tools_called: list[str]      # 최종적으로 답에 쓴 도구 집합 (채점 대상)
    trace: list[str]


# ---------------------------------------------------------------------------
# offline(키 없음) 대체 구현: 키워드 분류 + 추출형 답변
# ---------------------------------------------------------------------------
def offline_classify(question: str) -> dict:
    q = question.lower()
    if any(re.search(p, q) for p in OFFLINE_OTHER_SIGNALS):
        return {"category": "C0", "confidence": 0.9, "alternative": None, "reason": "범위 밖 신호 키워드"}
    hits = {k: sum(1 for w in words if w.lower() in q) for k, words in OFFLINE_KEYWORDS.items()}
    total = sum(hits.values())
    if total == 0:
        return {"category": "C0", "confidence": 0.6, "alternative": None, "reason": "어느 분야 키워드도 없음"}
    ranked = sorted(hits, key=lambda k: -hits[k])
    best, second = ranked[0], ranked[1]
    if hits[best] == hits[second]:  # 동점이면 각 카테고리 근거와의 유사도로 순서를 정한다
        secs = load_sections()
        sims = {k: max(similarity(question, [secs[s].body for s in CATEGORIES[k]["sections"]])) for k in (best, second)}
        best, second = sorted((best, second), key=lambda k: -sims[k])
    return {
        "category": best, "confidence": round(hits[best] / total, 2),
        "alternative": second if hits[second] else None, "reason": f"키워드 일치 {hits}",
    }


def offline_answer(question: str, sections: list[Section], max_sentences: int = 5, min_ratio: float = 0.35) -> str:
    cands = []
    for si, s in enumerate(sections):
        for li, line in enumerate(re.split(r"\n|(?<=다\.)\s+", s.text)):
            line = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
            if len(line) > 10 and not line.startswith(">"):
                cands.append((si, li, line, s.id))
    if not cands:
        return ""
    sims = similarity(question, [c[2] for c in cands])
    top = sorted(zip(cands, sims), key=lambda x: -x[1])[:max_sentences]
    top = [t for t in top if t[1] >= top[0][1] * min_ratio]  # 가장 관련 높은 문장 대비 너무 동떨어진 문장은 뺀다
    top = sorted(top, key=lambda x: (x[0][0], x[0][1]))
    return "\n".join(f"{c[2]} [{c[3]}]" for c, _ in top)


# ---------------------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------------------
def jev_classify(jev, question: str) -> dict:
    """Jev choice 질문으로 카테고리별 확률을 받는다. 신뢰도 = 1위 확률, 대안 = 2위(확률 0.1 이상일 때)."""
    ans = jev.decide({"question": question}, jev_router_questions(CATEGORIES))
    probs = ans["category"]["probabilities"]
    ranked = sorted(probs, key=lambda k: -probs[k])
    top, second = ranked[0], ranked[1]
    return {
        "category": top, "confidence": probs[top], "alternative": second if probs[second] >= 0.1 else None,
        "evidence_types": [t for i, t in enumerate(EVIDENCE_TYPES) if ans[f"etype_{i}"]["noul"] >= 0.5],
        "reason": "Jev 카테고리 확률 " + ", ".join(f"{k} {probs[k]:.2f}" for k in ranked[:3]),
        "probabilities": {k: round(probs[k], 3) for k in ranked},
    }


def build_graph(llm: LLM, settings: Settings, jev=None):
    use_jev = settings.decider == "jev" and jev is not None

    def classify(state: AgentState) -> AgentState:
        q = state["question"]
        if use_jev:
            out = jev_classify(jev, q)
        elif llm.online:
            system, user = router_prompt(q, CATEGORIES)
            out = llm.json(system, user)
        else:
            out = offline_classify(q)
        cat = out.get("category") if out.get("category") in CATEGORIES else "C0"
        alt = out.get("alternative") if out.get("alternative") in CATEGORIES and out.get("alternative") != cat else None
        try:
            conf = max(0.0, min(1.0, float(out.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0.0
        return {
            "category": cat, "confidence": conf, "alternative": alt,
            "evidence_types": out.get("evidence_types") or [], "category_reason": out.get("reason", ""),
            "probabilities": out.get("probabilities"),
            "trace": state.get("trace", []) + [f"① 판정: {cat} (신뢰도 {conf:.2f})"],
        }

    def check_confidence(state: AgentState) -> AgentState:
        # 판정과 분리된 판단: 신뢰도가 임계값보다 낮으면 되묻는다. C0은 되묻지 않고 ②에서 보류한다.
        low = state["category"] != "C0" and state["confidence"] < settings.confidence_threshold
        note = f"신뢰도 {state['confidence']:.2f} {'<' if low else '≥'} 임계값 {settings.confidence_threshold}"
        return {"trace": state["trace"] + [f"신뢰도 점검: {note} → {'되묻기' if low else '통과'}"]}

    def clarify(state: AgentState) -> AgentState:
        options = [state["category"]] + ([state["alternative"]] if state.get("alternative") else [])
        lines = "\n".join(f"- {CATEGORIES[k]['name']}({k}): {CATEGORIES[k]['focus']}" for k in options)
        text = ("[확인 필요] 질문이 어느 분야를 묻는지 분명하지 않아 답변 전에 확인이 필요합니다"
                f"(판정 신뢰도 {state['confidence']:.2f}).\n어느 관점의 설명이 필요한지 질문을 조금 더 구체적으로 적어 주세요.\n{lines}")
        return {"status": "clarify", "answer": text, "tools_called": [CLARIFY], "citations": [],
                "trace": state["trace"] + ["→ 되묻기"]}

    def gather(state: AgentState) -> AgentState:
        asm = assemble(state["question"], state["category"], settings, llm, jev if use_jev else None)
        msg = f"② 근거 조립: {asm.tool or '-'} → {[s.id for s in asm.sections]} / {'충분' if asm.sufficient else '보류'}"
        return {
            "tool": asm.tool, "sections": asm.sections, "candidates": asm.candidates, "scores": asm.scores, "gate": asm.gate,
            "sufficient": asm.sufficient, "handoff_reason": asm.handoff_reason, "attempts": 0, "failed_drafts": [],
            "trace": state["trace"] + [msg],
        }

    def answer(state: AgentState) -> AgentState:
        q, secs = state["question"], state["sections"]
        prev = state.get("verification")
        if llm.online:
            system, user = answer_prompt(q, secs, feedback(prev) if prev and not prev["passed"] else None)
            text = llm.complete(system, user).strip()
        else:
            text = offline_answer(q, secs)
        n = state.get("attempts", 0) + 1
        return {"answer": text, "attempts": n, "trace": state["trace"] + [f"③ 답변 생성 ({n}회차)"]}

    def check(state: AgentState) -> AgentState:
        result = verify(state["answer"], state["sections"], state["question"], state["category"])
        if use_jev:  # 기계적 검사에 더해, Jev가 문장마다 근거 지지 확률을 낸다
            c = jev_support_check(state["answer"], state["sections"], jev, settings.jev_support_threshold)
            result["checks"].append(c)
            result["passed"] = result["passed"] and c["passed"]
        failed = state.get("failed_drafts", [])
        if not result["passed"]:
            failed = failed + [{"answer": state["answer"], "problems": feedback(result)}]
        return {"verification": result, "failed_drafts": failed,
                "trace": state["trace"] + [f"④ 검증: {'통과' if result['passed'] else '실패'}"]}

    def abstain(state: AgentState) -> AgentState:
        cat = CATEGORIES[state["category"]]
        reason = state.get("handoff_reason") or "근거가 부족합니다."
        if state.get("verification") and not state["verification"]["passed"]:
            # 실패 내용(근거 밖 수치·용어)은 지어낸 내용 자체이므로 사용자 문구에 넣지 않는다. 화면의 ④ 검증 패널에만 보인다.
            reason = "근거만으로 검증을 통과하는 답변을 만들지 못했습니다."
        text = f"[보류] 학습노트 근거로는 답할 수 없어 답변을 보류합니다.\n사유: {reason}\n확인할 곳: {cat['handoff_to']}"
        return {"status": "abstained", "answer": text, "handoff_reason": reason, "tools_called": [ABSTAIN],
                "citations": [], "trace": state["trace"] + ["→ 보류"]}

    def finalize(state: AgentState) -> AgentState:
        return {"status": "answered", "tools_called": [state["tool"]], "citations": citations(state["answer"]),
                "trace": state["trace"] + ["→ 답변 완료"]}

    def after_confidence(state: AgentState) -> str:
        low = state["category"] != "C0" and state["confidence"] < settings.confidence_threshold
        return "clarify" if low else "gather"

    def after_gather(state: AgentState) -> str:
        return "answer" if state["sufficient"] else "abstain"

    def after_check(state: AgentState) -> str:
        if state["verification"]["passed"]:
            return "finalize"
        return "answer" if state["attempts"] < MAX_ANSWER_ATTEMPTS and llm.online else "abstain"

    g = StateGraph(AgentState)
    for name, fn in [("classify", classify), ("check_confidence", check_confidence), ("clarify", clarify),
                     ("gather", gather), ("answer", answer), ("verify", check), ("abstain", abstain),
                     ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_edge("classify", "check_confidence")
    g.add_conditional_edges("check_confidence", after_confidence, {"clarify": "clarify", "gather": "gather"})
    g.add_conditional_edges("gather", after_gather, {"answer": "answer", "abstain": "abstain"})
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", after_check, {"finalize": "finalize", "answer": "answer", "abstain": "abstain"})
    for end in ("clarify", "abstain", "finalize"):
        g.add_edge(end, END)
    return g.compile()


class Agent:
    def __init__(self, provider: str | None = None, settings: Settings | None = None):
        self.llm = LLM(provider)
        self.settings = settings or Settings()
        self.jev = None
        if self.settings.decider == "jev":
            from jev import Jev

            self.jev = Jev()
            if not self.jev.available:
                raise RuntimeError("decider=jev인데 .env에 Jev 키·주소가 없습니다(JEV_API_KEY, 또는 LAYA_API_KEY + LAYA_BASE_URL).")
        self.graph = build_graph(self.llm, self.settings, self.jev)

    def run(self, question: str) -> AgentState:
        return self.graph.invoke({"question": question, "trace": []})


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    agent = Agent()
    q = " ".join(sys.argv[1:]) or "현금전환주기가 뭐야?"
    out = agent.run(q)
    print(f"[{agent.llm.provider}] " + " | ".join(out["trace"]))
    print(out["answer"])
    for c in out.get("citations", []):
        print(f"  출처 [{c['id']}] {c['doc']} · {c['title']} — {c['source']}")
    if out.get("verification"):
        for c in out["verification"]["checks"]:
            print(("  ✓ " if c["passed"] else "  ✗ ") + f"{c['name']}: {c['detail']}")
