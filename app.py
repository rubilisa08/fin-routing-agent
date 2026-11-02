"""FinRoute AI 데모 화면: python -m streamlit run app.py"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from context import CATEGORIES, Settings, load_sections
from llm import resolve_provider
from verify import CITE_RE, ID_RE

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

EXAMPLES = [
    "미지급 이자는 결산 때 어떻게 분개해?",
    "재무상태표의 자본은 어떤 항목으로 구성돼?",
    "이자보상배율이 1보다 작으면 무슨 뜻이야?",
    "회수기간법의 단점은?",
    "부채는 많을수록 나빠?",
    "이연법인세자산은 언제 인식해?",
    "비트코인 내년에 오를까?",
]
STATUS = {"answered": ("답변", "✅"), "abstained": ("보류", "↪️"), "clarify": ("되묻기", "❓")}

st.set_page_config(page_title="FinRoute AI", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="에이전트 준비 중…")
def get_agent(provider: str, knobs: tuple):
    from agent import Agent

    keys = ("top_k", "min_similarity", "confidence_threshold", "use_scope_rules", "use_term_coverage", "use_llm_gate")
    return Agent(provider, Settings(**dict(zip(keys, knobs))))


def cite_badges(text: str) -> str:
    return CITE_RE.sub(lambda m: " " + " ".join(f"`{sid}`" for sid in ID_RE.findall(m.group(0))), text)


# ---------------------------------------------------------------------------
# 사이드바: 실행 모드와 실험 손잡이
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    detected = resolve_provider()
    options = ["gemini", "openai", "offline"]
    provider = st.selectbox("LLM", options, index=options.index(detected) if detected in options else 2,
                            help="API 키가 없으면 offline(키워드 분류 + 추출형 답변)으로 동작합니다.")
    if provider != detected and detected == "offline":
        st.warning(".env에 API 키가 없어 이 모드는 실패할 수 있습니다.")
    conf = st.slider("판정 신뢰도 임계값 (이보다 낮으면 되묻기)", 0.0, 1.0, Settings.confidence_threshold, 0.05)
    top_k = st.slider("근거 섹션 수 (top_k)", 2, 8, Settings.top_k)
    use_scope = st.toggle("범위 규칙으로 보류", True)
    use_term = st.toggle("용어 커버리지로 보류", True)
    use_gate = st.toggle("LLM 근거 충분성 판단", True, disabled=provider == "offline")
    st.divider()
    st.caption("예시 질문")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    st.divider()
    st.caption("입력 → ① 판정(신뢰도) → 신뢰도 점검(낮으면 되묻기) → ② 근거 조립(부족하면 보류) → ③ 답변 → ④ 검증")

st.title("📊 FinRoute AI")
st.caption("재무·회계 질문을 4개 분야로 분류하고, 그 분야의 근거 문서에서만 답합니다. "
           "분야가 모호하면 되묻고, 근거가 없으면 지어내지 않고 보류합니다.")

tab_ask, tab_eval, tab_map = st.tabs(["질문하기", "평가 결과", "카테고리·근거 매핑"])

# ---------------------------------------------------------------------------
with tab_ask:
    with st.form("ask"):
        q = st.text_input("질문", key="q", placeholder="예: 듀폰분석이 뭐야?")
        submitted = st.form_submit_button("질문하기", type="primary")

    if submitted and q.strip():
        agent = get_agent(provider, (top_k, Settings.min_similarity, conf, use_scope, use_term, use_gate))
        try:
            with st.spinner("판정 → 근거 조립 → 답변 → 검증 중…"):
                out = agent.run(q.strip())
        except Exception as e:  # 키 오류·쿼터 초과 등
            st.error(f"실행 실패: {e}")
            st.stop()

        cat = CATEGORIES[out["category"]]
        status_label, icon = STATUS[out["status"]]
        ver = out.get("verification")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("판정 카테고리", f"{out['category']} {cat['name']}")
        c2.metric("판정 신뢰도", f"{out['confidence']:.2f}")
        c3.metric("결과", f"{status_label} · {', '.join(out['tools_called'])}")
        c4.metric("검증", "해당 없음" if not ver or out["status"] == "clarify" else ("통과" if ver["passed"] else "실패"))

        # 생성된 답변과 검색된 근거(출처)를 나눠서 보여 준다
        if out["status"] == "answered":
            st.success(cite_badges(out["answer"]).replace("\n", "  \n"), icon=icon)
            st.markdown("**출처**")
            st.dataframe(pd.DataFrame([{
                "섹션": c["id"], "문서": c["doc"], "제목": c["title"], "출처": c["source"], "버전": c["version"],
            } for c in out.get("citations", [])]), hide_index=True, use_container_width=True)
        elif out["status"] == "clarify":
            st.info(out["answer"].replace("\n", "  \n"), icon=icon)
        else:
            st.warning(out["answer"].replace("\n", "  \n"), icon=icon)

        st.subheader("처리 과정")
        s1, s2 = st.columns(2)
        with s1:
            with st.container(border=True):
                st.markdown(f"**① 카테고리 판정** — `{out['category']}` {cat['name']} ({cat['en']})")
                st.progress(out["confidence"], text=f"신뢰도 {out['confidence']:.2f} / 임계값 {conf:.2f}")
                if out.get("alternative"):
                    st.caption(f"두 번째 후보: {out['alternative']} {CATEGORIES[out['alternative']]['name']}")
                if out.get("evidence_types"):
                    st.caption("필요한 근거 유형: " + ", ".join(out["evidence_types"]))
                st.caption(out.get("category_reason") or "")
            with st.container(border=True):
                st.markdown(f"**② 근거 조립** — 도구 `{out.get('tool') or '없음'}`")
                if out["status"] == "clarify":
                    st.caption("신뢰도가 낮아 근거를 조회하지 않고 되물었습니다.")
                for g in out.get("gate", []):
                    st.markdown(f"{'✅' if g['passed'] else '⛔'} `{g['check']}` {g['detail']}")
                if out.get("sections") and out["status"] == "answered":
                    st.markdown("프롬프트에 넣은 근거 섹션")
                    for s in out["sections"]:
                        with st.expander(f"[{s.id}] {s.title}  ·  유사도 {out['scores'].get(s.id, 0):.3f}"):
                            st.caption(f"{s.doc_title} · 출처: {s.source} · {s.version}")
                            st.markdown(s.text)
                elif out["status"] == "abstained":
                    st.caption(f"보류 사유: {out.get('handoff_reason')}")
        with s2:
            with st.container(border=True):
                st.markdown(f"**③ 답변** — {out.get('attempts', 0)}회 생성")
                st.caption("근거 섹션만 프롬프트에 넣고, 문장마다 섹션 ID를 붙이게 합니다.")
                for i, d in enumerate(out.get("failed_drafts") or [], 1):
                    with st.expander(f"검증에서 걸러진 초안 {i}"):
                        st.markdown(cite_badges(d["answer"]))
                        st.caption(" / ".join(d["problems"]))
            with st.container(border=True):
                st.markdown("**④ 검증 (기계적 검사)**")
                if ver:
                    for c in ver["checks"]:
                        st.markdown(f"{'✅' if c['passed'] else '❌'} **{c['name']}** — {c['detail']}")
                else:
                    st.caption("보류·되묻기는 답변을 만들지 않으므로 검증하지 않습니다.")
        st.caption(" → ".join(out["trace"]))

# ---------------------------------------------------------------------------
with tab_eval:
    hist = RESULTS / "history.md"
    if hist.exists():
        st.markdown("#### 개선 시도별 기록")
        st.markdown(hist.read_text(encoding="utf-8"))
        runs = sorted(RESULTS.glob("run_*.json"), reverse=True)
        pick = st.selectbox("실행 결과 보기", runs, format_func=lambda p: p.stem)
        if pick:
            data = json.loads(pick.read_text(encoding="utf-8"))
            sm = data["summary"]
            fmt = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731
            st.caption("과제 지표")
            a1, a2 = st.columns(2)
            a1.metric("도구 호출 적절성", fmt(sm["tool_score"]))
            a2.metric("답변 적절성", fmt(sm["answer_score"]))
            st.caption("PRD 지표")
            m = st.columns(5)
            m[0].metric("분류 정확도", fmt(sm["router_accuracy"]))
            m[1].metric("근거 적합성", fmt(sm.get("retrieval_relevance")))
            m[2].metric("답변 정확성", fmt(sm.get("answer_accuracy")))
            m[3].metric("근거 없는 정보 생성", fmt(sm.get("unsupported_rate")))
            m[4].metric("보류 정확도", fmt(sm.get("abstain_accuracy")))
            df = pd.DataFrame([{
                "ID": r["id"], "질문": r["question"], "판정": f"{r['category']} {r.get('confidence') or 0:.2f}",
                "기대 도구": ", ".join(r["expected_tools"]), "실제 도구": ", ".join(r["tools_called"]),
                "도구": "○" if r["tool_score"] else "×", "답변": "○" if r["answer_score"] else "×",
                "원인": " / ".join(r["diagnosis"]),
            } for r in data["records"]])
            st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("아직 평가 기록이 없습니다. `python evaluate.py --label baseline`을 실행하세요.")

# ---------------------------------------------------------------------------
with tab_map:
    secs = load_sections()
    rows = []
    for key, c in CATEGORIES.items():
        rows.append({
            "카테고리": f"{key} · {c['name']} ({c['en']})", "도구": c["tool"] or "abstain",
            "근거 섹션": ", ".join(f"{s} {secs[s].title}" for s in c["sections"]) or "(없음 → 보류)",
            "다루지 않는 범위(→ 보류)": c["out_of_scope"], "넘길 곳": c["handoff_to"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("#### 섹션별 출처")
    st.dataframe(pd.DataFrame([{"섹션": s.id, "제목": s.title, "문서": s.doc_title, "출처": s.source, "버전": s.version}
                               for s in secs.values()]), hide_index=True, use_container_width=True)
