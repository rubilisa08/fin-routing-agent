"""FinRoute AI 데모 화면: python -m streamlit run app.py"""
from __future__ import annotations

import html
import json
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
# 결과별 표시 이름과 색 이름(CSS 클래스)
STATUS = {"answered": ("답변", "ok"), "abstained": ("보류", "warn"), "clarify": ("되묻기", "info")}

st.set_page_config(page_title="FinRoute AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
:root {
  --fr-accent: #2f6fed; --fr-ok: #15803d; --fr-warn: #b45309; --fr-info: #1d4ed8; --fr-bad: #b91c1c;
  --fr-line: rgba(127,127,127,.22); --fr-soft: rgba(127,127,127,.06); --fr-muted: rgba(127,127,127,.95);
}
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { font-size: 1.7rem !important; }
.fr-sub { font-size: 13.5px; color: var(--fr-muted); margin: -6px 0 14px; }
.fr-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 4px 0 14px; }
.fr-kpi { border: 1px solid var(--fr-line); border-radius: 10px; padding: 10px 12px; background: var(--fr-soft); min-width: 0; }
.fr-kpi .l { font-size: 12px; color: var(--fr-muted); margin-bottom: 4px; }
.fr-kpi .v { font-size: 15px; font-weight: 600; line-height: 1.35; overflow-wrap: anywhere; }
.fr-kpi .s { font-size: 12px; color: var(--fr-muted); margin-top: 3px; overflow-wrap: anywhere; }
.fr-pill { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 12.5px; font-weight: 600; }
.fr-pill.ok { background: rgba(21,128,61,.13); color: var(--fr-ok); }
.fr-pill.warn { background: rgba(180,83,9,.13); color: var(--fr-warn); }
.fr-pill.info { background: rgba(29,78,216,.12); color: var(--fr-info); }
.fr-pill.bad { background: rgba(185,28,28,.12); color: var(--fr-bad); }
.fr-pill.muted { background: var(--fr-soft); color: var(--fr-muted); border: 1px solid var(--fr-line); }
.fr-answer { border: 1px solid var(--fr-line); border-left: 4px solid var(--fr-c); border-radius: 10px;
  padding: 12px 16px; background: var(--fr-soft); font-size: 14.5px; line-height: 1.75; margin-bottom: 10px; }
.fr-answer.ok { --fr-c: var(--fr-ok); } .fr-answer.warn { --fr-c: var(--fr-warn); } .fr-answer.info { --fr-c: var(--fr-info); }
.fr-cite { font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; padding: 1px 6px; margin: 0 1px;
  border-radius: 6px; background: rgba(47,111,237,.12); color: var(--fr-accent); white-space: nowrap; }
.fr-h { font-size: 13px; font-weight: 600; margin: 4px 0 6px; }
.fr-src { font-size: 13px; padding: 6px 0; border-top: 1px dashed var(--fr-line); }
.fr-src .m { color: var(--fr-muted); font-size: 12px; }
.fr-step { font-size: 14px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.fr-num { display: inline-flex; width: 21px; height: 21px; border-radius: 50%; background: var(--fr-accent); color: #fff;
  font-size: 12px; align-items: center; justify-content: center; flex: none; }
.fr-note { font-size: 12.5px; color: var(--fr-muted); line-height: 1.6; margin: 4px 0; }
.fr-bar { position: relative; height: 8px; border-radius: 4px; background: var(--fr-line); margin: 8px 0 4px; }
.fr-bar .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: var(--fr-accent); }
.fr-bar .mark { position: absolute; top: -4px; width: 2px; height: 16px; background: currentColor; opacity: .75; }
.fr-scale { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--fr-muted); }
.fr-chip { display: inline-block; font-size: 11.5px; padding: 1px 7px; border-radius: 6px; margin: 2px 3px 0 0;
  border: 1px solid var(--fr-line); }
.fr-cand { display: grid; grid-template-columns: 62px 1fr 92px 46px; gap: 8px; align-items: center; font-size: 12.5px;
  padding: 4px 0; border-top: 1px dashed var(--fr-line); }
.fr-cand.dim { opacity: .55; }
.fr-mini { height: 6px; border-radius: 3px; background: var(--fr-line); position: relative; }
.fr-mini .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--fr-accent); }
.fr-check { font-size: 13px; padding: 4px 0; border-top: 1px dashed var(--fr-line); line-height: 1.5; }
.fr-check b { font-weight: 600; } .fr-check .x { color: var(--fr-bad); font-weight: 700; } .fr-check .o { color: var(--fr-ok); font-weight: 700; }
.fr-trace { font-size: 12px; color: var(--fr-muted); margin-top: 6px; }
@media (max-width: 760px) { .fr-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } .fr-cand { grid-template-columns: 58px 1fr 46px; } .fr-cand .mini { display: none; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="에이전트 준비 중…")
def get_agent(provider: str, knobs: tuple):
    from agent import Agent

    keys = ("top_k", "min_similarity", "confidence_threshold", "use_scope_rules", "use_term_coverage", "use_llm_gate", "decider")
    return Agent(provider, Settings(**dict(zip(keys, knobs))))


def esc(text) -> str:
    return html.escape(str(text))


def rich(text: str) -> str:
    """LLM이 쓴 글을 안전하게 HTML로: 이스케이프 → 인용 [AP-07]을 배지로 → 줄바꿈."""
    out = CITE_RE.sub(lambda m: "".join(f'<span class="fr-cite">{sid}</span>' for sid in ID_RE.findall(m.group(0))),
                      esc(text))
    return out.replace("\n", "<br>")


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="fr-kpi"><div class="l">{label}</div><div class="v">{value}</div>' + \
        (f'<div class="s">{sub}</div>' if sub else "") + "</div>"


def pill(text: str, kind: str) -> str:
    return f'<span class="fr-pill {kind}">{esc(text)}</span>'


def html_block(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 사이드바: 실행 모드와 실험 손잡이
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("설정")
    detected = resolve_provider()
    options = ["openai", "gemini", "offline"]
    provider = st.selectbox("답변 LLM", options, index=options.index(detected) if detected in options else 2,
                            help="API 키가 없으면 offline(키워드 분류 + 추출형 답변)으로 동작합니다.")
    if provider != detected and detected == "offline":
        st.warning(".env에 API 키가 없어 이 모드는 실패할 수 있습니다.")
    decider = st.radio("판정·충분성·근거 지지 판단", ["llm", "jev"], horizontal=True, disabled=provider == "offline",
                       help="jev: 의사결정 모델이 보정된 확률로 판단합니다(.env에 Jev 키 필요). 답변 작성은 LLM이 합니다.")
    conf = st.slider("판정 신뢰도 임계값", 0.0, 1.0, Settings.confidence_threshold, 0.05,
                     help="판정 신뢰도가 이 값보다 낮으면 근거를 찾기 전에 되묻습니다. C0(지원하지 않는 질문)은 되묻지 않고 보류합니다.")
    top_k = st.slider("검색 후보 섹션 수 (top_k)", 2, 8, Settings.top_k,
                      help="판정된 분야의 섹션 중 질문과 가장 비슷한 k개를 후보로 뽑습니다. 근거 충분성 판단이 그중 필요한 섹션만 골라 답변에 씁니다.")
    use_scope = st.toggle("범위 규칙으로 보류", True)
    use_term = st.toggle("용어 커버리지로 보류", True)
    use_gate = st.toggle("근거 충분성 판단(게이트)", True, disabled=provider == "offline")
    st.divider()
    st.caption("예시 질문")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state["q"] = ex

st.title("📊 FinRoute AI")
html_block('<div class="fr-sub">재무·회계 질문을 4개 분야로 분류하고 그 분야의 근거 문서에서만 답합니다. '
           '분야가 모호하면 되묻고, 근거가 없으면 지어내지 않고 보류합니다.</div>')

tab_ask, tab_eval, tab_map = st.tabs(["질문하기", "평가 결과", "카테고리·근거 매핑"])

# ---------------------------------------------------------------------------
with tab_ask:
    with st.form("ask"):
        q = st.text_input("질문", key="q", placeholder="예: 듀폰분석이 뭐야?")
        submitted = st.form_submit_button("질문하기", type="primary")

    if submitted and q.strip():
        agent = get_agent(provider, (top_k, Settings.min_similarity, conf, use_scope, use_term, use_gate,
                                     decider if provider != "offline" else "llm"))
        try:
            with st.spinner("판정 → 근거 조립 → 답변 → 검증 중…"):
                out = agent.run(q.strip())
        except Exception as e:  # 키 오류·쿼터 초과 등
            st.error(f"실행 실패: {e}")
            st.stop()

        cat = CATEGORIES[out["category"]]
        status_label, kind = STATUS[out["status"]]
        ver = out.get("verification")
        low = out["category"] != "C0" and out["confidence"] < conf

        # 요약 카드 4개
        if not ver or out["status"] != "answered":
            ver_value, ver_sub = pill("해당 없음", "muted"), "보류·되묻기는 검증하지 않음"
        else:
            n_ok = sum(c["passed"] for c in ver["checks"])
            ver_value = pill("통과" if ver["passed"] else "실패", "ok" if ver["passed"] else "bad")
            ver_sub = f"검사 {len(ver['checks'])}개 중 {n_ok}개 통과 · 답변 {out.get('attempts', 0)}회 생성"
        html_block('<div class="fr-kpis">'
                   + kpi("판정 카테고리", f"{out['category']} · {esc(cat['name'])}", esc(cat["en"]))
                   + kpi("판정 신뢰도", f"{out['confidence']:.2f}",
                         f"임계값 {conf:.2f} → " + ("C0은 되묻지 않음" if out["category"] == "C0" else ("되묻기" if low else "통과")))
                   + kpi("결과", pill(status_label, kind), "도구 " + esc(", ".join(out["tools_called"])))
                   + kpi("검증", ver_value, ver_sub)
                   + "</div>")

        # 생성된 답변과 검색된 근거(출처)를 나눠서 보여 준다
        html_block(f'<div class="fr-answer {kind}">{rich(out["answer"])}</div>')
        if out["status"] == "answered" and out.get("citations"):
            html_block('<div class="fr-h">출처</div>' + "".join(
                f'<div class="fr-src"><span class="fr-cite">{c["id"]}</span> <b>{esc(c["title"])}</b> '
                f'<span class="m">· {esc(c["doc"])} · {esc(c["source"])} · {esc(c["version"])}</span></div>'
                for c in out["citations"]))

        st.write("")
        st.markdown("##### 처리 과정")
        s1, s2 = st.columns(2, gap="medium")
        with s1:
            with st.container(border=True):
                pct = out["confidence"] * 100
                body = (f'<div class="fr-step"><span class="fr-num">1</span>카테고리 판정 · {out["category"]} {esc(cat["name"])}</div>'
                        f'<div class="fr-bar"><div class="fill" style="width:{pct:.0f}%"></div>'
                        f'<div class="mark" style="left:calc({conf * 100:.0f}% - 1px)" title="임계값"></div></div>'
                        f'<div class="fr-scale"><span>신뢰도 {out["confidence"]:.2f}</span><span>│ 임계값 {conf:.2f}</span></div>')
                if out.get("probabilities"):
                    body += '<div class="fr-note">Jev 카테고리 확률</div>' + "".join(
                        f'<div class="fr-cand"><span>{k}</span><div class="fr-mini mini"><div class="fill" style="width:{p * 100:.0f}%"></div></div>'
                        f'<span></span><span>{p:.2f}</span></div>' for k, p in out["probabilities"].items())
                if out.get("alternative"):
                    alt = out["alternative"]
                    body += f'<div class="fr-note">두 번째 후보: {alt} {esc(CATEGORIES[alt]["name"])}</div>'
                if out.get("evidence_types"):
                    body += '<div class="fr-note">필요한 근거 유형 ' + "".join(
                        f'<span class="fr-chip">{esc(t)}</span>' for t in out["evidence_types"]) + "</div>"
                if out.get("category_reason"):
                    body += f'<div class="fr-note">{esc(out["category_reason"])}</div>'
                html_block(body)

            with st.container(border=True):
                body = f'<div class="fr-step"><span class="fr-num">2</span>근거 조립 · {esc(out.get("tool") or "조회 안 함")}</div>'
                if out["status"] == "clarify":
                    body += '<div class="fr-note">신뢰도가 임계값보다 낮아 근거를 조회하지 않고 되물었습니다.</div>'
                for g in out.get("gate", []):
                    mark = '<span class="o">✓</span>' if g["passed"] else '<span class="x">✕</span>'
                    body += f'<div class="fr-check">{mark} <b>{esc(g["check"])}</b> — {esc(g["detail"])}</div>'
                cands = out.get("candidates") or []
                if cands:
                    used = {s.id for s in out.get("sections") or []} if out["status"] == "answered" else set()
                    top = max(out["scores"].values(), default=1) or 1
                    secs = load_sections()
                    body += f'<div class="fr-note" style="margin-top:8px">검색 후보 top_k={len(cands)} · 유사도 순</div>' + "".join(
                        f'<div class="fr-cand{"" if sid in used else " dim"}"><span class="fr-cite">{sid}</span>'
                        f'<span>{esc(secs[sid].title)}</span>'
                        f'<div class="fr-mini mini"><div class="fill" style="width:{out["scores"][sid] / top * 100:.0f}%"></div></div>'
                        f'<span>{pill("사용", "ok") if sid in used else f"{out["scores"][sid]:.2f}"}</span></div>'
                        for sid in cands)
                if out["status"] == "abstained":
                    body += f'<div class="fr-note">보류 사유: {esc(out.get("handoff_reason"))}</div>'
                html_block(body)
                if out["status"] == "answered":
                    for s in out.get("sections") or []:
                        with st.expander(f"[{s.id}] {s.title} 원문 보기"):
                            st.caption(f"{s.doc_title} · 출처: {s.source} · {s.version}")
                            st.markdown(s.text)

        with s2:
            with st.container(border=True):
                n = out.get("attempts", 0)
                note = ("근거 섹션만 프롬프트에 넣고 문장마다 섹션 ID를 붙이게 합니다." if n else
                        "보류·되묻기라 답변을 만들지 않았습니다.")
                html_block(f'<div class="fr-step"><span class="fr-num">3</span>답변 생성 · {n}회</div><div class="fr-note">{note}</div>')
                for i, d in enumerate(out.get("failed_drafts") or [], 1):
                    with st.expander(f"검증에서 걸러진 초안 {i}"):
                        html_block(f'<div class="fr-note">{rich(d["answer"])}</div>'
                                   + "".join(f'<div class="fr-check"><span class="x">✕</span> {esc(p)}</div>' for p in d["problems"]))
            with st.container(border=True):
                body = '<div class="fr-step"><span class="fr-num">4</span>검증 · 기계적 검사</div>'
                if ver and out["status"] == "answered":
                    for c in ver["checks"]:
                        mark = '<span class="o">✓</span>' if c["passed"] else '<span class="x">✕</span>'
                        body += f'<div class="fr-check">{mark} <b>{esc(c["name"])}</b> — {esc(c["detail"])}</div>'
                else:
                    body += '<div class="fr-note">보류·되묻기는 답변을 만들지 않으므로 검증하지 않습니다.</div>'
                html_block(body)
        html_block(f'<div class="fr-trace">{esc(" → ".join(out["trace"]))}</div>')

# ---------------------------------------------------------------------------
with tab_eval:
    hist = RESULTS / "history.md"
    if hist.exists():
        runs = sorted(RESULTS.glob("run_*.json"), reverse=True)
        pick = st.selectbox("실행 결과 보기", runs, format_func=lambda p: p.stem)
        if pick:
            data = json.loads(pick.read_text(encoding="utf-8"))
            sm = data["summary"]
            fmt = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731
            html_block('<div class="fr-h">과제 지표</div><div class="fr-kpis">'
                       + kpi("도구 호출 적절성", fmt(sm["tool_score"]))
                       + kpi("답변 적절성", fmt(sm["answer_score"]))
                       + kpi("모델", esc(f"{sm['provider']}/{sm['model']}"), esc(sm.get("data", "")))
                       + kpi("채점기", esc(sm["grader"]))
                       + '</div><div class="fr-h">PRD 지표</div><div class="fr-kpis" style="grid-template-columns:repeat(5,minmax(0,1fr))">'
                       + kpi("분류 정확도", fmt(sm["router_accuracy"]))
                       + kpi("근거 적합성", fmt(sm.get("retrieval_relevance")))
                       + kpi("답변 정확성", fmt(sm.get("answer_accuracy")))
                       + kpi("근거 없는 정보 생성", fmt(sm.get("unsupported_rate")))
                       + kpi("보류 정확도", fmt(sm.get("abstain_accuracy")))
                       + "</div>")
            df = pd.DataFrame([{
                "ID": r["id"], "질문": r["question"], "판정": f"{r['category']} {r.get('confidence') or 0:.2f}",
                "기대 도구": ", ".join(r["expected_tools"]), "실제 도구": ", ".join(r["tools_called"]),
                "도구": "○" if r["tool_score"] else "×", "답변": "○" if r["answer_score"] else "×",
                "원인": " / ".join(r["diagnosis"]),
            } for r in data["records"]])
            st.dataframe(df, hide_index=True, width="stretch")
        with st.expander("개선 시도별 기록 전체"):
            st.markdown(hist.read_text(encoding="utf-8"))
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
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("##### 섹션별 출처")
    st.dataframe(pd.DataFrame([{"섹션": s.id, "제목": s.title, "문서": s.doc_title, "출처": s.source, "버전": s.version}
                               for s in secs.values()]), hide_index=True, width="stretch")
