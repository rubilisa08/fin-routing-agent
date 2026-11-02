"""파이프라인: ① 카테고리 판정 → ② 근거 조립(넘기기 판단) → ③ 근거만으로 답변 → ④ 검증.

LangGraph StateGraph로 잇는다. 검증 실패 시 한 번 다시 답하고, 그래도 실패하면 넘긴다.
"""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from context import CATEGORIES, HANDOFF, Section, Settings, assemble, similarity
from llm import LLM
from prompts import OFFLINE_KEYWORDS, OFFLINE_OTHER_SIGNALS, answer_prompt, router_prompt
from verify import feedback, verify

MAX_ANSWER_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    question: str
    category: str                # ① 판정 결과
    category_reason: str
    tool: str | None             # ② 조회한 근거 도구
    sections: list[Section]      # ② 프롬프트에 넣은 근거 섹션
    scores: dict[str, float]     # ② 카테고리 안 섹션별 유사도
    gate: list[dict]             # ② 넘기기 판단 기록
    sufficient: bool
    handoff_reason: str
    answer: str                  # ③
    attempts: int
    verification: dict           # ④
    status: str                  # "answered" | "handoff"
    tools_called: list[str]      # 최종적으로 답에 쓴 도구 집합 (채점 대상)
    trace: list[str]


# ---------------------------------------------------------------------------
# offline(키 없음) 대체 구현: 키워드 분류 + 추출형 답변
# ---------------------------------------------------------------------------
def offline_classify(question: str) -> tuple[str, str]:
    q = question.lower()
    if any(re.search(p, q) for p in OFFLINE_OTHER_SIGNALS):
        return "other", "범위 밖 신호 키워드"
    hits = {k: sum(1 for w in words if w.lower() in q) for k, words in OFFLINE_KEYWORDS.items()}
    best = max(hits.values())
    if best == 0:
        return "other", "어느 분야 키워드도 없음"
    tied = [k for k, v in hits.items() if v == best]
    if len(tied) > 1:  # 동점이면 각 카테고리 근거와의 유사도로 가른다
        from context import load_sections

        secs = load_sections()
        sims = {k: max(similarity(question, [secs[s].body for s in CATEGORIES[k]["sections"]])) for k in tied}
        tied = [max(sims, key=sims.get)]
    return tied[0], f"키워드 일치 {hits}"


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
def build_graph(llm: LLM, settings: Settings):
    def classify(state: AgentState) -> AgentState:
        q = state["question"]
        if llm.online:
            system, user = router_prompt(q, CATEGORIES)
            out = llm.json(system, user)
            cat = out.get("category", "other")
            cat = cat if cat in CATEGORIES else "other"
            reason = out.get("reason", "")
        else:
            cat, reason = offline_classify(q)
        return {"category": cat, "category_reason": reason, "trace": state.get("trace", []) + [f"① 판정: {cat}"]}

    def gather(state: AgentState) -> AgentState:
        asm = assemble(state["question"], state["category"], settings, llm)
        msg = f"② 근거 조립: {asm.tool or '-'} → {[s.id for s in asm.sections]} / {'충분' if asm.sufficient else '넘김'}"
        return {
            "tool": asm.tool, "sections": asm.sections, "scores": asm.scores, "gate": asm.gate,
            "sufficient": asm.sufficient, "handoff_reason": asm.handoff_reason, "attempts": 0,
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
        result = verify(state["answer"], state["sections"], state["question"])
        return {"verification": result, "trace": state["trace"] + [f"④ 검증: {'통과' if result['passed'] else '실패'}"]}

    def handoff(state: AgentState) -> AgentState:
        cat = CATEGORIES[state["category"]]
        reason = state.get("handoff_reason") or "근거가 부족합니다."
        if state.get("verification") and not state["verification"]["passed"]:
            # 실패 내용(근거 밖 수치·용어)은 지어낸 내용 자체이므로 사용자 문구에 넣지 않는다. 화면의 ④ 검증 패널에만 보인다.
            reason = "근거만으로 검증을 통과하는 답변을 만들지 못했습니다."
        text = f"[넘김] 학습노트 근거로는 답할 수 없어 답변을 보류합니다.\n사유: {reason}\n확인할 곳: {cat['handoff_to']}"
        return {"status": "handoff", "answer": text, "handoff_reason": reason, "tools_called": [HANDOFF],
                "trace": state["trace"] + ["→ 넘김"]}

    def finalize(state: AgentState) -> AgentState:
        return {"status": "answered", "tools_called": [state["tool"]], "trace": state["trace"] + ["→ 답변 완료"]}

    def after_gather(state: AgentState) -> str:
        return "answer" if state["sufficient"] else "handoff"

    def after_check(state: AgentState) -> str:
        if state["verification"]["passed"]:
            return "finalize"
        return "answer" if state["attempts"] < MAX_ANSWER_ATTEMPTS and llm.online else "handoff"

    g = StateGraph(AgentState)
    for name, fn in [("classify", classify), ("gather", gather), ("answer", answer),
                     ("verify", check), ("handoff", handoff), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_edge("classify", "gather")
    g.add_conditional_edges("gather", after_gather, {"answer": "answer", "handoff": "handoff"})
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", after_check, {"finalize": "finalize", "answer": "answer", "handoff": "handoff"})
    g.add_edge("handoff", END)
    g.add_edge("finalize", END)
    return g.compile()


class Agent:
    def __init__(self, provider: str | None = None, settings: Settings | None = None):
        self.llm = LLM(provider)
        self.settings = settings or Settings()
        self.graph = build_graph(self.llm, self.settings)

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
    if out.get("verification"):
        for c in out["verification"]["checks"]:
            print(("  ✓ " if c["passed"] else "  ✗ ") + f"{c['name']}: {c['detail']}")
