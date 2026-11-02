"""데모 화면: streamlit run app.py"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from context import CATEGORIES, Settings, load_sections
from llm import resolve_provider

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

EXAMPLES = [
    "재무상태표의 자본은 어떤 항목으로 구성돼?",
    "이자보상배율이 1보다 작으면 무슨 뜻이야?",
    "핵심감사사항이 뭐야?",
    "금리가 오르면 채권 가격은 어떻게 돼?",
    "이연법인세자산은 언제 인식해?",
    "비트코인 내년에 오를까?",
]

st.set_page_config(page_title="재무·회계 학습 라우팅 에이전트", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="에이전트 준비 중…")
def get_agent(provider: str, knobs: tuple):
    from agent import Agent

    keys = ("top_k", "min_similarity", "use_scope_rules", "use_term_coverage", "use_llm_gate")
    return Agent(provider, Settings(**dict(zip(keys, knobs))))


def cite_badges(text: str) -> str:
    return re.sub(r"\[([A-Z]{2}-\d{2})\]", r" `\1`", text)


# ---------------------------------------------------------------------------
# 사이드바: 실행 모드와 실험 손잡이
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    detected = resolve_provider()
    provider = st.selectbox("LLM", ["gemini", "openai", "offline"], index=["gemini", "openai", "offline"].index(detected),
                            help="API 키가 없으면 offline(키워드 분류 + 추출형 답변)으로 동작합니다.")
    if provider != detected and detected == "offline":
        st.warning(".env에 API 키가 없어 이 모드는 실패할 수 있습니다.")
    top_k = st.slider("근거 섹션 수 (top_k)", 2, 8, Settings.top_k)
    use_scope = st.toggle("범위 규칙으로 넘기기", True)
    use_term = st.toggle("용어 커버리지로 넘기기", True)
    use_gate = st.toggle("LLM 근거 충분성 판단", True, disabled=provider == "offline")
    st.divider()
    st.caption("예시 질문")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    st.divider()
    st.caption("입력 → ① 카테고리 판정 → ② 근거 조립(모르면 넘김) → ③ 근거만으로 답변 → ④ 검증")

st.title("📊 재무·회계 학습 라우팅 에이전트")
st.caption("질문을 분야별로 분류하고, 그 분야의 학습노트에서만 근거를 찾아 답합니다. 근거가 없으면 지어내지 않고 넘깁니다.")

tab_ask, tab_eval, tab_map = st.tabs(["질문하기", "평가 결과", "카테고리·근거 매핑"])

# ---------------------------------------------------------------------------
with tab_ask:
    with st.form("ask"):
        q = st.text_input("질문", key="q", placeholder="예: 듀폰분석이 뭐야?")
        submitted = st.form_submit_button("질문하기", type="primary")

    if submitted and q.strip():
        agent = get_agent(provider, (top_k, Settings.min_similarity, use_scope, use_term, use_gate))
        try:
            with st.spinner("판정 → 근거 조립 → 답변 → 검증 중…"):
                out = agent.run(q.strip())
        except Exception as e:  # 키 오류·쿼터 초과 등
            st.error(f"실행 실패: {e}")
            st.stop()

        cat = CATEGORIES[out["category"]]
        handed_off = out["status"] == "handoff"

        # 최종 결과
        c1, c2, c3 = st.columns(3)
        c1.metric("판정 카테고리", cat["name"])
        c2.metric("호출한 도구", ", ".join(out["tools_called"]))
        ver = out.get("verification")
        c3.metric("검증", "해당 없음(넘김)" if handed_off and not ver else ("통과" if ver and ver["passed"] else "실패"))

        if handed_off:
            st.warning(out["answer"].replace("\n", "  \n"), icon="↪️")
        else:
            st.success(cite_badges(out["answer"]).replace("\n", "  \n"), icon="✅")

        st.subheader("처리 과정")
        s1, s2 = st.columns(2)
        with s1:
            with st.container(border=True):
                st.markdown(f"**① 카테고리 판정** — `{out['category']}` {cat['name']}")
                st.caption(out.get("category_reason") or "")
            with st.container(border=True):
                st.markdown(f"**② 근거 조립** — 도구 `{out.get('tool') or '없음'}`")
                for g in out.get("gate", []):
                    st.markdown(f"{'✅' if g['passed'] else '⛔'} `{g['check']}` {g['detail']}")
                if out.get("sections") and not handed_off:
                    st.markdown("프롬프트에 넣은 근거 섹션")
                    for s in out["sections"]:
                        with st.expander(f"[{s.id}] {s.title}  ·  유사도 {out['scores'].get(s.id, 0):.3f}"):
                            st.markdown(s.text)
                elif handed_off:
                    st.caption(f"넘김 사유: {out.get('handoff_reason')}")
        with s2:
            with st.container(border=True):
                st.markdown(f"**③ 답변** — {out.get('attempts', 0)}회 생성")
                st.caption("근거 섹션만 프롬프트에 넣고, 문장마다 섹션 ID를 붙이게 합니다.")
            with st.container(border=True):
                st.markdown("**④ 검증 (기계적 검사)**")
                if ver:
                    for c in ver["checks"]:
                        st.markdown(f"{'✅' if c['passed'] else '❌'} **{c['name']}** — {c['detail']}")
                    for w in ver["warnings"]:
                        st.markdown(f"⚠️ {w}")
                else:
                    st.caption("넘긴 질문은 답변을 만들지 않으므로 검증하지 않습니다.")
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
            m1, m2, m3 = st.columns(3)
            m1.metric("도구 호출 적절성", f"{sm['tool_score']:.0%}")
            m2.metric("답변 적절성", f"{sm['answer_score']:.0%}")
            m3.metric("① 판정 정확도(참고)", f"{sm['router_accuracy']:.0%}")
            df = pd.DataFrame([{
                "ID": r["id"], "질문": r["question"], "기대 도구": ", ".join(r["expected_tools"]),
                "실제 도구": ", ".join(r["tools_called"]), "도구": "○" if r["tool_score"] else "×",
                "답변": "○" if r["answer_score"] else "×", "원인": " / ".join(r["diagnosis"]),
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
            "카테고리": f"{key} · {c['name']}", "도구": c["tool"] or "handoff",
            "근거 섹션": ", ".join(f"{s} {secs[s].title}" for s in c["sections"]) or "(없음 → 넘김)",
            "다루지 않는 범위(→ 넘김)": c["out_of_scope"], "넘길 곳": c["handoff_to"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
