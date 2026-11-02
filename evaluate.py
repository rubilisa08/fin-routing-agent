"""두 지표 측정: 도구 호출 적절성 · 답변 적절성.

    python evaluate.py --validate-grader          # 모범/오답 답안으로 채점기 자체를 검증
    python evaluate.py --label baseline           # 평가셋 전체 실행 → results/에 기록
    python evaluate.py --label no-term --no-term-coverage   # 손잡이 하나만 바꿔 다시 측정

결과는 results/run_<시각>_<label>.json, results/run_<...>_errors.md, results/history.md(누적 기록표)에 남는다.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from context import CATEGORIES, HANDOFF, Settings, similarity
from llm import LLM, QuotaExhausted, resolve_provider
from prompts import fewshot, judge_prompt

ROOT = Path(__file__).parent
GOLDEN = ROOT / "data" / "goldenset.json"
RESULTS = ROOT / "results"


def load_items(ids: list[str] | None = None, path: Path = GOLDEN) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    return [it for it in items if not ids or it["id"] in ids]


def check_split(items: list[dict]) -> list[str]:
    """예시용(fewshot)과 평가용(goldenset) 질문이 겹치거나 거의 같은지 검사."""
    shots = [e["question"] for e in fewshot()["router"]] + [e["question"] for e in fewshot()["sufficiency"]]
    problems = []
    for it in items:
        sims = similarity(it["question"], shots)
        best = max(range(len(shots)), key=lambda i: sims[i])
        if sims[best] > 0.6:
            problems.append(f"{it['id']} ≈ 예시 '{shots[best]}' (유사도 {sims[best]:.2f})")
    return problems


def is_handoff(answer: str) -> bool:
    return answer.lstrip().startswith("[넘김]")


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
    missing = [f["fact"] for f in out.get("facts", []) if not f.get("included")]
    violated = [v["claim"] for v in out.get("violations", []) if v.get("violated")]
    return {"missing": missing, "violations": violated, "judge": out}


def grade_answer(item: dict, answer: str, grader: str, llm: LLM | None) -> dict:
    expects_handoff = item["expected_tools"] == [HANDOFF]
    if expects_handoff or is_handoff(answer):
        # 넘김 경로는 형식이 정해져 있어 기계적으로 본다
        if expects_handoff != is_handoff(answer):
            why = "넘겨야 하는데 답을 지어냄" if expects_handoff else "답할 수 있는데 넘김"
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
        mark = "✓" if r[4] else "✗"
        line = f" {mark} {r[0]} {r[1]:<11} 기대 {r[2]} → {r[3]}"
        if not r[4]:
            line += f"  누락={r[5]} 위반={r[6]}"
        print(line)
    return ok == total


# ---------------------------------------------------------------------------
# 평가 실행
# ---------------------------------------------------------------------------
def diagnose(item: dict, out: dict, grade: dict) -> list[str]:
    """틀린 건의 원인을 적는다: 어느 근거를 잘못 골랐는지, 어떤 사실을 빠뜨렸는지."""
    notes = []
    exp, got = item["expected_tools"], out["tools_called"]
    if exp != got:
        if exp == [HANDOFF]:
            notes.append(f"[도구] 넘겨야 하는데 {got}로 답함 → 근거 밖 내용을 지어낼 위험")
        elif got == [HANDOFF]:
            failed = [g for g in out.get("gate", []) if not g["passed"]]
            ver = out.get("verification")
            if out["category"] != item["category"]:
                cause = f"① 판정 오류: '{out['category']}'(기대 '{item['category']}')로 분류되어 넘김"
            elif ver and not ver["passed"]:
                cause = "검증 실패 후 넘김: " + "; ".join(c["detail"] for c in ver["checks"] if not c["passed"])
            elif failed:
                cause = f"② 게이트 '{failed[0]['check']}'에서 넘김: {failed[0]['detail']}"
            else:
                cause = "원인 미상"
            notes.append(f"[도구] 답할 수 있는데 넘김 — {cause}")
        else:
            notes.append(f"[도구] 다른 근거 조회: 기대 {exp}, 실제 {got} (① 판정 '{out['category']}', 기대 '{item['category']}')")
    if item["evidence"] and out.get("sections") and got != [HANDOFF]:
        used = [s.id for s in out["sections"]]
        lacking = [e for e in item["evidence"] if e not in used]
        if lacking:
            notes.append(f"[근거] 필요한 섹션 {lacking}이 조립되지 않음 (조립: {used})")
    for m in grade["missing"]:
        notes.append(f"[답변] 필수 사실 누락: {m}")
    for v in grade["violations"]:
        notes.append(f"[답변] 금지 위반: {v}")
    return notes


def run_eval(args) -> None:
    from agent import Agent

    settings = Settings(
        top_k=args.top_k, min_similarity=args.min_sim, use_scope_rules=not args.no_scope_rules,
        use_term_coverage=not args.no_term_coverage, use_llm_gate=not args.no_llm_gate,
    )
    agent = Agent(args.provider, settings)
    grader = args.grader or ("llm" if agent.llm.online else "keyword")
    judge = agent.llm if grader == "llm" else None
    items = load_items(args.ids, args.data)

    records = []
    for it in items:
        try:
            out = agent.run(it["question"])
        except QuotaExhausted as e:
            # 지금까지의 호출은 .llm_cache/에 남아 있으므로 내일 같은 명령을 다시 실행하면 이어서 진행된다
            print(f"\n⏸ {e}\n{len(records)}/{len(items)}문항까지 진행. 같은 명령을 다시 실행하면 이어서 합니다 "
                  f"(API 호출 {agent.llm.api_calls}회, 캐시 {agent.llm.cache_hits}회).")
            return
        tool_ok = int(sorted(out["tools_called"]) == sorted(it["expected_tools"]))
        grade = grade_answer(it, out["answer"], grader, judge)
        rec = {
            "id": it["id"], "question": it["question"],
            "expected_category": it["category"], "category": out["category"],
            "expected_tools": it["expected_tools"], "tools_called": out["tools_called"],
            "tool_score": tool_ok, "answer_score": grade["score"],
            "sections": [s.id for s in out.get("sections") or []],
            "gate": out.get("gate", []), "verification": out.get("verification"),
            "answer": out["answer"], "missing": grade["missing"], "violations": grade["violations"],
            "diagnosis": diagnose(it, out, grade) if not (tool_ok and grade["score"]) else [],
        }
        records.append(rec)
        print(f"{it['id']} 도구 {'○' if tool_ok else '×'} 답변 {'○' if grade['score'] else '×'}  "
              f"[{out['category']}] {out['tools_called']}")

    n = len(records)
    tool = sum(r["tool_score"] for r in records)
    ans = sum(r["answer_score"] for r in records)
    route = sum(r["category"] == r["expected_category"] for r in records)
    summary = {
        "label": args.label, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "provider": agent.llm.provider, "model": agent.llm.model, "grader": grader,
        "settings": settings.as_dict(), "n": n,
        "tool_score": tool / n, "answer_score": ans / n, "router_accuracy": route / n,
        "api_calls": agent.llm.api_calls if agent.llm.online else 0,
        "cache_hits": agent.llm.cache_hits if agent.llm.online else 0,
    }
    print(f"\n도구 호출 적절성 {tool}/{n} = {tool / n:.0%} | 답변 적절성 {ans}/{n} = {ans / n:.0%} "
          f"| (참고) ① 판정 정확도 {route}/{n} | API 호출 {summary['api_calls']}회, 캐시 {summary['cache_hits']}회")

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
        hist.write_text("| 시각 | 라벨 | 모델 | 채점기 | 설정 | 도구 호출 | 답변 | ① 판정 |\n|---|---|---|---|---|---|---|---|\n",
                        encoding="utf-8")
    s = settings
    knobs = f"top_k={s.top_k}, scope={int(s.use_scope_rules)}, term={int(s.use_term_coverage)}, gate={int(s.use_llm_gate)}, sim={s.min_similarity}"
    with hist.open("a", encoding="utf-8") as f:
        f.write(f"| {summary['time']} | {args.label} | {agent.llm.provider}/{agent.llm.model} | {grader} | {knobs} "
                f"| {tool}/{n} ({tool / n:.0%}) | {ans}/{n} ({ans / n:.0%}) | {route}/{n} |\n")
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
    p.add_argument("--no-scope-rules", action="store_true")
    p.add_argument("--no-term-coverage", action="store_true")
    p.add_argument("--no-llm-gate", action="store_true")
    args = p.parse_args()

    items = load_items(args.ids, args.data)
    split = check_split(items)
    print("예시/평가 분리 점검:", "겹침 없음" if not split else split)
    counts = {k: sum(it["category"] == k for it in items) for k in CATEGORIES}
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
