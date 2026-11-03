"""FinRoute AI 측정.

과제 필수 지표 2개(도구 호출 적절성 · 답변 적절성)와 PRD 지표 5개를 함께 잰다.

    python evaluate.py --validate-grader          # 모범/오답 답안으로 채점기 자체를 검증
    python evaluate.py --label baseline           # 평가셋 전체 실행 → results/에 기록
    python evaluate.py --label conf0.7 --conf 0.7 # 손잡이 하나만 바꿔 다시 측정

결과는 results/run_<시각>_<label>.json, results/run_<...>_errors.md, results/history.md(누적 기록표)에 남는다.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from context import ABSTAIN, CATEGORIES, CLARIFY, Settings, similarity
from llm import LLM, QuotaExhausted, resolve_provider
from prompts import fewshot, judge_prompt

ROOT = Path(__file__).parent
GOLDEN = ROOT / "data" / "goldenset.json"
HOLDOUT = ROOT / "data" / "holdout.json"
RESULTS = ROOT / "results"


def load_items(ids: list[str] | None = None, path: Path = GOLDEN) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    return [it for it in items if not ids or it["id"] in ids]


def check_split(items: list[dict], path: Path) -> list[str]:
    """예시용(fewshot)과 평가용 질문, 평가셋과 미공개셋 질문이 겹치거나 거의 같은지 검사."""
    shots = [e["question"] for e in fewshot()["router"]] + [e["question"] for e in fewshot()["sufficiency"]]
    others = [it["question"] for it in load_items(path=HOLDOUT if Path(path) == GOLDEN else GOLDEN)]
    problems = []
    for it in items:
        for label, pool in (("예시", shots), ("다른 평가 파일", others)):
            sims = similarity(it["question"], pool)
            best = max(range(len(pool)), key=lambda i: sims[i])
            if sims[best] > 0.6:
                problems.append(f"{it['id']} ≈ {label} '{pool[best]}' (유사도 {sims[best]:.2f})")
    return problems


def outcome(answer: str) -> str:
    a = answer.lstrip()
    if a.startswith(("[보류]", "[넘김]")):
        return ABSTAIN
    if a.startswith("[확인 필요]"):
        return CLARIFY
    return "answer"


def expected_outcome(item: dict) -> str:
    t = item["expected_tools"]
    return t[0] if t in ([ABSTAIN], [CLARIFY]) else "answer"


def category_ok(item: dict, category: str) -> bool:
    return category in (item.get("acceptable_categories") or [item["category"]])


# ---------------------------------------------------------------------------
# 채점기
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).replace("−", "-").replace("–", "-").replace("—", "-").casefold()


def violations_by_pattern(item: dict, answer: str) -> list[str]:
    return [c["claim"] for c in item["must_not"] if any(re.search(p, answer, re.I) for p in c["patterns"])]


def grade_keyword(item: dict, answer: str) -> dict:
    """키 없이 쓰는 채점기: 사실마다 키워드 묶음(묶음 안은 OR, 묶음끼리는 AND)이 모두 맞아야 담은 것으로 본다."""
    a = _norm(answer)
    missing = [f["fact"] for f in item["must_include"]
               if not all(any(_norm(k) in a for k in group) for group in f["keywords"])]
    return {"missing": missing, "violations": violations_by_pattern(item, answer)}


def grade_llm(item: dict, answer: str, llm: LLM) -> dict:
    """LLM 채점기: 표현이 아니라 사실을 본다(prompts.JUDGE_SYSTEM)."""
    facts = [f["fact"] for f in item["must_include"]]
    claims = [c["claim"] for c in item["must_not"]]
    system, user = judge_prompt(item["question"], answer, facts, claims)
    out = llm.json(system, user)
    # 판정은 문항 순서대로 맞춘다. 형식이 어긋나면(문자열 목록, 개수 불일치) 키워드 채점기로 대신하고 그 사실을 기록한다.
    f_out, v_out = out.get("facts", []), out.get("violations", [])
    well_formed = (len(f_out) == len(facts) and len(v_out) == len(claims)
                   and all(isinstance(x, dict) for x in f_out + v_out))
    if not well_formed:
        g = grade_keyword(item, answer)
        g["judge"] = {"fallback": "keyword", "raw": out}
        return g
    missing = [facts[i] for i, f in enumerate(f_out) if not f.get("included")]
    violated = [claims[i] for i, v in enumerate(v_out) if v.get("violated")]
    return {"missing": missing, "violations": violated, "judge": out}


def grade_answer(item: dict, answer: str, grader: str, llm: LLM | None) -> dict:
    exp, got = expected_outcome(item), outcome(answer)
    if exp != "answer" or got != "answer":
        # 보류·되묻기 경로는 형식이 정해져 있어 기계적으로 본다
        if exp != got:
            why = {(ABSTAIN, "answer"): "보류해야 하는데 답을 지어냄", (CLARIFY, "answer"): "되물어야 하는데 답함",
                   ("answer", ABSTAIN): "답할 수 있는데 보류", ("answer", CLARIFY): "답할 수 있는데 되물음"}.get(
                (exp, got), f"기대 {exp}, 실제 {got}")
            return {"score": 0, "missing": [why], "violations": violations_by_pattern(item, answer)}
        v = violations_by_pattern(item, answer)
        return {"score": int(not v), "missing": [], "violations": v}
    g = grade_llm(item, answer, llm) if grader == "llm" else grade_keyword(item, answer)
    g["score"] = int(not g["missing"] and not g["violations"])
    return g


def validate_grader(items: list[dict], grader: str, llm: LLM | None) -> bool:
    """모범 답안은 1점, 오답은 0점이 나와야 채점기를 믿을 수 있다."""
    ok = 0
    rows = []
    for it in items:
        for kind, expected in (("gold_answer", 1), ("bad_answer", 0)):
            g = grade_answer(it, it[kind], grader, llm)
            hit = g["score"] == expected
            ok += hit
            rows.append((it["id"], kind, expected, g["score"], hit, g["missing"], g["violations"]))
    total = len(rows)
    print(f"\n채점기 검증 ({grader}): {ok}/{total} 일치")
    for r in rows:
        if not r[4]:
            print(f" ✗ {r[0]} {r[1]:<11} 기대 {r[2]} → {r[3]}  누락={r[5]} 위반={r[6]}")
    return ok == total


# ---------------------------------------------------------------------------
# 평가 실행
# ---------------------------------------------------------------------------
def diagnose(item: dict, out: dict, grade: dict) -> list[str]:
    """틀린 건의 원인을 적는다: 어느 근거를 잘못 골랐는지, 어떤 사실을 빠뜨렸는지."""
    notes = []
    exp, got = item["expected_tools"], out["tools_called"]
    if not category_ok(item, out["category"]):
        notes.append(f"[판정] '{out['category']}'(신뢰도 {out.get('confidence', 0):.2f})로 분류, 기대 "
                     f"{item.get('acceptable_categories') or item['category']}")
    if exp != got:
        if exp == [ABSTAIN]:
            notes.append(f"[도구] 보류해야 하는데 {got} → 근거 밖 내용을 지어낼 위험")
        elif exp == [CLARIFY]:
            notes.append(f"[도구] 되물어야 하는데 {got} (신뢰도 {out.get('confidence', 0):.2f})")
        elif got == [CLARIFY]:
            notes.append(f"[도구] 답할 수 있는데 되물음 (신뢰도 {out.get('confidence', 0):.2f})")
        elif got == [ABSTAIN]:
            failed = [g for g in out.get("gate", []) if not g["passed"]]
            ver = out.get("verification")
            if ver and not ver["passed"]:
                cause = "검증 2회 실패 후 보류: " + "; ".join(c["detail"] for c in ver["checks"] if not c["passed"])
            elif failed:
                cause = f"② 게이트 '{failed[0]['check']}'에서 보류: {failed[0]['detail']}"
            else:
                cause = "① 판정이 C0"
            notes.append(f"[도구] 답할 수 있는데 보류 — {cause}")
        else:
            notes.append(f"[도구] 다른 근거 조회: 기대 {exp}, 실제 {got}")
    if item["evidence"] and out.get("sections") and got not in ([ABSTAIN], [CLARIFY]):
        used = [s.id for s in out["sections"]]
        lacking = [e for e in item["evidence"] if e not in used]
        if lacking:
            notes.append(f"[근거] 필요한 섹션 {lacking}이 조립되지 않음 (조립: {used})")
    for d in out.get("failed_drafts") or []:
        notes.append(f"[검증] 걸러진 초안: {'; '.join(d['problems'])}")
    for m in grade["missing"]:
        notes.append(f"[답변] 필수 사실 누락: {m}")
    for v in grade["violations"]:
        notes.append(f"[답변] 금지 위반: {v}")
    return notes


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def run_eval(args) -> None:
    from agent import Agent

    settings = Settings(
        top_k=args.top_k, min_similarity=args.min_sim, confidence_threshold=args.conf,
        use_scope_rules=not args.no_scope_rules, use_term_coverage=not args.no_term_coverage,
        use_llm_gate=not args.no_llm_gate, decider=args.decider,
        jev_gate_threshold=args.jev_gate, jev_support_threshold=args.jev_support,
    )
    agent = Agent(args.provider, settings)
    grader = args.grader or ("llm" if agent.llm.online else "keyword")
    judge = agent.llm if grader == "llm" else None
    items = load_items(args.ids, args.data)

    records = []
    for it in items:
        try:
            out = agent.run(it["question"])
            grade = grade_answer(it, out["answer"], grader, judge)
        except QuotaExhausted as e:
            # 지금까지의 호출은 .llm_cache/에 남아 있으므로 같은 명령을 다시 실행하면 이어서 진행된다
            print(f"\n⏸ {e}\n{len(records)}/{len(items)}문항까지 진행. 같은 명령을 다시 실행하면 이어서 합니다 "
                  f"(API 호출 {agent.llm.api_calls}회, 캐시 {agent.llm.cache_hits}회).")
            return
        tool_ok = int(sorted(out["tools_called"]) == sorted(it["expected_tools"]))
        exp_out, got_out = expected_outcome(it), outcome(out["answer"])
        used = [s.id for s in out.get("sections") or []]
        rec = {
            "id": it["id"], "question": it["question"],
            "expected_category": it.get("acceptable_categories") or [it["category"]], "category": out["category"],
            "confidence": out.get("confidence"), "probabilities": out.get("probabilities"),
            "category_ok": int(category_ok(it, out["category"])),
            "expected_tools": it["expected_tools"], "tools_called": out["tools_called"], "tool_score": tool_ok,
            "expected_outcome": exp_out, "outcome": got_out, "outcome_ok": int(exp_out == got_out),
            "evidence": it["evidence"], "sections": used,
            "retrieval_ok": int(all(e in used for e in it["evidence"])) if it["evidence"] else None,
            "answer_score": grade["score"], "missing": grade["missing"], "violations": grade["violations"],
            "unsupported": int(got_out == "answer" and bool(grade["violations"])),
            "failed_drafts": len(out.get("failed_drafts") or []),
            "gate": out.get("gate", []), "verification": out.get("verification"),
            "answer": out["answer"], "citations": out.get("citations", []),
            "diagnosis": diagnose(it, out, grade) if not (tool_ok and grade["score"]) else [],
        }
        records.append(rec)
        print(f"{it['id']} 도구 {'○' if tool_ok else '×'} 답변 {'○' if grade['score'] else '×'}  "
              f"[{out['category']} {out.get('confidence', 0):.2f}] {out['tools_called']}")

    n = len(records)
    answerable = [r for r in records if r["expected_outcome"] == "answer"]
    answered = [r for r in records if r["outcome"] == "answer"]
    with_ev = [r for r in records if r["retrieval_ok"] is not None]
    m = {
        "tool": sum(r["tool_score"] for r in records),
        "answer": sum(r["answer_score"] for r in records),
        "route": sum(r["category_ok"] for r in records),
        "retrieval": sum(r["retrieval_ok"] for r in with_ev),
        "faithful": sum(r["answer_score"] for r in answerable),
        "unsupported": sum(r["unsupported"] for r in answered),
        "abstain": sum(r["outcome_ok"] for r in records),
        "caught": sum(1 for r in records if r["failed_drafts"]),
    }
    summary = {
        "label": args.label, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "provider": agent.llm.provider, "model": agent.llm.model, "grader": grader,
        "settings": settings.as_dict(), "n": n, "data": Path(args.data).name,
        "tool_score": m["tool"] / n, "answer_score": m["answer"] / n,
        "router_accuracy": m["route"] / n,
        "retrieval_relevance": m["retrieval"] / len(with_ev) if with_ev else None,
        "answer_accuracy": m["faithful"] / len(answerable) if answerable else None,
        "unsupported_rate": m["unsupported"] / len(answered) if answered else 0.0,
        "abstain_accuracy": m["abstain"] / n,
        "drafts_caught_by_verifier": m["caught"],
        "api_calls": agent.llm.api_calls if agent.llm.online else 0,
        "cache_hits": agent.llm.cache_hits if agent.llm.online else 0,
        "jev_model": agent.jev.last_model if agent.jev else None,
        "jev_calls": agent.jev.api_calls if agent.jev else 0,
        "jev_cache_hits": agent.jev.cache_hits if agent.jev else 0,
    }
    print(f"\n[과제] 도구 호출 적절성 {pct(m['tool'], n)} | 답변 적절성 {pct(m['answer'], n)}")
    print(f"[PRD] ① 분류 정확도 {pct(m['route'], n)} | ② 근거 적합성 {pct(m['retrieval'], len(with_ev))} | "
          f"③ 답변 정확성 {pct(m['faithful'], len(answerable))} | ④ 근거 없는 정보 생성 {pct(m['unsupported'], len(answered))} | "
          f"⑤ 보류 정확도 {pct(m['abstain'], n)}")
    print(f"검증이 걸러낸 초안이 있던 문항 {m['caught']}개 | LLM 호출 {summary['api_calls']}회(캐시 {summary['cache_hits']}회)"
          + (f" | Jev 호출 {summary['jev_calls']}회(캐시 {summary['jev_cache_hits']}회), 모델 {summary['jev_model']}" if agent.jev else ""))

    RESULTS.mkdir(exist_ok=True)
    stem = f"run_{datetime.now():%Y%m%d_%H%M%S}_{args.label}"
    (RESULTS / f"{stem}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    errs = [f"# 오답 분석 — {args.label}\n", f"{summary['time']} · {summary['provider']}/{summary['model']} · 채점 {grader}\n"]
    for r in records:
        if r["diagnosis"]:
            errs.append(f"\n## {r['id']} {r['question']}\n- 기대 도구 {r['expected_tools']} / 실제 {r['tools_called']}\n")
            errs += [f"- {d}\n" for d in r["diagnosis"]]
            errs.append(f"\n```\n{r['answer']}\n```\n")
    (RESULTS / f"{stem}_errors.md").write_text("".join(errs), encoding="utf-8")

    hist = RESULTS / "history.md"
    if not hist.exists():
        hist.write_text(
            "| 시각 | 라벨 | 데이터 | 모델 | 채점기 | 설정 | 도구 호출 | 답변 적절성 | 분류 | 근거 적합 | 답변 정확 | 근거 없는 생성 | 보류 정확 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    s = settings
    knobs = (f"decider={s.decider}, top_k={s.top_k}, conf={s.confidence_threshold}, scope={int(s.use_scope_rules)}, "
             f"term={int(s.use_term_coverage)}, gate={int(s.use_llm_gate)}"
             + (f", jev_gate={s.jev_gate_threshold}, jev_support={s.jev_support_threshold}" if s.decider == "jev" else ""))
    with hist.open("a", encoding="utf-8") as f:
        f.write(f"| {summary['time']} | {args.label} | {summary['data']} | {agent.llm.provider}/{agent.llm.model} | {grader} "
                f"| {knobs} | {pct(m['tool'], n)} | {pct(m['answer'], n)} | {pct(m['route'], n)} "
                f"| {pct(m['retrieval'], len(with_ev))} | {pct(m['faithful'], len(answerable))} "
                f"| {pct(m['unsupported'], len(answered))} | {pct(m['abstain'], n)} |\n")
    print(f"기록: results/{stem}.json, results/{stem}_errors.md, results/history.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate-grader", action="store_true")
    p.add_argument("--label", default="run")
    p.add_argument("--provider", default=None, help="gemini | openai | offline (기본: 키를 보고 자동)")
    p.add_argument("--grader", choices=["keyword", "llm"], default=None)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--data", default=str(GOLDEN), help="평가 파일 (기본 data/goldenset.json, 과적합 점검은 data/holdout.json)")
    p.add_argument("--top-k", type=int, default=Settings.top_k)
    p.add_argument("--min-sim", type=float, default=Settings.min_similarity)
    p.add_argument("--conf", type=float, default=Settings.confidence_threshold, help="판정 신뢰도 임계값(이보다 낮으면 되묻기)")
    p.add_argument("--no-scope-rules", action="store_true")
    p.add_argument("--no-term-coverage", action="store_true")
    p.add_argument("--no-llm-gate", action="store_true")
    p.add_argument("--decider", choices=["llm", "jev"], default=Settings.decider, help="판정·충분성·근거 지지 판단에 쓸 모델")
    p.add_argument("--jev-gate", type=float, default=Settings.jev_gate_threshold)
    p.add_argument("--jev-support", type=float, default=Settings.jev_support_threshold)
    args = p.parse_args()

    items = load_items(args.ids, args.data)
    split = check_split(items, args.data)
    print("예시/평가 분리 점검:", "겹침 없음" if not split else split)
    counts = {k: sum(it["category"] == k for it in items) for k in list(CATEGORIES) + ["ambiguous"]}
    print("카테고리별 문항 수:", counts)

    if args.validate_grader:
        provider = args.provider or resolve_provider()
        grader = args.grader or ("llm" if provider != "offline" else "keyword")
        validate_grader(items, grader, LLM(provider) if grader == "llm" else None)
        return
    run_eval(args)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()
