"""분류 지침 · 근거 충분성 판단 · 답변 규칙 · 채점 지침.

예시는 data/fewshot.json에서만 가져온다(평가셋과 분리).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FEWSHOT_PATH = Path(__file__).parent / "data" / "fewshot.json"


@lru_cache(maxsize=1)
def fewshot() -> dict:
    return json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ① 카테고리 판정: 카테고리만 고른다. 넘길지 말지는 여기서 판단하지 않는다.
# ---------------------------------------------------------------------------
def router_prompt(question: str, categories: dict) -> tuple[str, str]:
    cat_lines = "\n".join(f"- {k}: {c['name']} — {c['description']}" for k, c in categories.items())
    examples = "\n".join(
        f"질문: {e['question']}\n출력: {json.dumps({'category': e['category']}, ensure_ascii=False)}"
        for e in fewshot()["router"]
    )
    system = f"""너는 재무·회계 학습 질문을 주제별로 분류하는 분류기다.
질문 하나를 아래 카테고리 중 정확히 하나로 분류한다.

{cat_lines}

규칙:
- 질문이 '무엇에 관한 지식'을 묻는지로 고른다. 질문에 나온 단어가 아니라 질문의 의도를 본다.
- 답을 알고 있는지, 근거가 있는지는 판단하지 않는다. 그 판단은 다음 단계가 한다.
- 개별 종목 매수·매도 추천, 주가 전망, 세금 신고, 특정 회사의 실적 수치 조회, 무관한 질문은 other다.
- 개념·공식·읽는 법을 묻는 질문은 네 분야 중 하나다. 그 분야 안에서 세부 수치(업종 평균, 현재 시장 수치)를 묻더라도 분야는 그대로 고른다.
- JSON으로만 답한다: {{"category": "<키>", "reason": "<한 문장>"}}

예시:
{examples}"""
    return system, f"질문: {question}"


# offline(키 없음) 분류 지침: 분야별 키워드. 가장 많이 겹치는 분야를 고른다.
OFFLINE_KEYWORDS = {
    "fs": ["재무제표", "재무상태표", "손익계산서", "포괄손익", "현금흐름표", "자본변동표", "주석", "유동자산",
           "비유동", "자산", "부채", "자본", "연결", "별도", "발생기준", "계속기업", "회계", "감가상각", "리스"],
    "ratio": ["비율", "roe", "roa", "듀폰", "이자보상", "회전율", "현금전환주기", "ccc", "증가율", "이익률",
              "레버리지", "잉여현금흐름", "fcf", "자본잠식"],
    "disclosure": ["공시", "dart", "다트", "사업보고서", "분기보고서", "반기보고서", "감사보고서", "감사의견",
                   "한정", "부적정", "의견거절", "핵심감사사항", "주요사항", "정정", "거래소", "상장", "kind"],
    "valuation": ["per", "pbr", "psr", "ebitda", "dcf", "wacc", "capm", "베타", "할인율", "현재가치", "미래가치",
                  "복리", "이자율", "금리", "채권", "듀레이션", "eps", "주당", "배당", "가치평가", "시장위험프리미엄"],
}
OFFLINE_OTHER_SIGNALS = [r"사도 ?될까|살까|팔까|추천|전망|오를까|내릴까", r"소득세|부가세|세금|신고", r"엑셀|vlookup|코인|점심"]


# ---------------------------------------------------------------------------
# ② 근거 충분성(넘기기 판단): 고른 카테고리의 근거만으로 답할 수 있는가
# ---------------------------------------------------------------------------
def sufficiency_prompt(question: str, category: dict, sections: list) -> tuple[str, str]:
    ex = "\n\n".join(
        f"질문: {e['question']}\n근거 섹션: {', '.join(e['sections'])}\n출력: {json.dumps(e['verdict'], ensure_ascii=False)}"
        for e in fewshot()["sufficiency"]
    )
    system = f"""너는 근거 충분성 심사관이다. 아래 근거 섹션'만'으로 질문에 답할 수 있는지 판단한다.
너의 일반 지식으로 보충해야만 답할 수 있다면 충분하지 않다.

판단 기준:
- 질문이 요구하는 핵심 정보(정의, 공식, 기준, 기한, 수치)가 근거에 명시되어 있어야 sufficient=true.
- 질문이 특정 수치(업종 평균, 현재 시장 수치, 특정 회사 실적)나 근거에 없는 규정을 요구하면 sufficient=false.
- 이 카테고리({category['name']})가 다루지 않는 범위: {category['out_of_scope']}
- needed_sections에는 답변에 실제로 필요한 섹션 ID만 넣는다. 관련 없는 섹션은 빼라.
- JSON으로만 답한다: {{"sufficient": true|false, "needed_sections": ["ID", ...], "reason": "<한 문장>", "missing": "<부족한 정보, 충분하면 빈 문자열>"}}

예시:
{ex}"""
    body = "\n\n".join(s.body for s in sections)
    return system, f"질문: {question}\n\n근거 섹션:\n{body}"


# ---------------------------------------------------------------------------
# ③ 근거만으로 답변
# ---------------------------------------------------------------------------
ANSWER_RULES = """너는 재무·회계 학습 도우미다. 아래 '근거 섹션'에 적힌 내용만으로 답한다.

답변 규칙:
1. 근거에 없는 사실·수치·용어·예시를 덧붙이지 않는다. 너의 일반 지식으로 보충하지 않는다.
2. 모든 문장 끝에 근거 섹션 ID를 [FS-01]처럼 붙인다. 근거 섹션 목록에 있는 ID만 쓴다.
3. 수치는 근거에 적힌 그대로 쓴다. 질문에 숫자가 주어진 계산 문제라면 근거의 공식에 그 숫자를 넣은 계산식을 보여 준다.
4. 질문의 일부를 근거가 다루지 않으면, 그 부분은 '학습노트 범위 밖'이라고 밝히고 추측하지 않는다.
5. 특정 종목의 매수·매도를 권하거나 주가를 전망하지 않는다.
6. 질문이 잘못된 전제(예: '무조건', '항상')를 깔고 있으면 근거에 따라 바로잡는다.
7. 한국어로 3~7문장, 필요하면 짧은 목록을 쓴다."""


def answer_prompt(question: str, sections: list, feedback: list[str] | None = None) -> tuple[str, str]:
    ex = "\n\n".join(f"질문: {e['question']}\n답변: {e['answer']}" for e in fewshot()["answer"])
    system = f"{ANSWER_RULES}\n\n예시:\n{ex}"
    body = "\n\n".join(s.body for s in sections)
    user = f"근거 섹션:\n{body}\n\n질문: {question}"
    if feedback:
        user += "\n\n이전 답변이 검증에 실패했다. 다음 문제를 고쳐서 다시 답하라:\n- " + "\n- ".join(feedback)
    return system, user


# ---------------------------------------------------------------------------
# 채점기(LLM judge): 표현이 아니라 사실을 본다
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """너는 엄격한 채점관이다. 답변이 '반드시 담아야 할 사실'을 담았는지, '말하면 안 되는 것'을 말했는지 판정한다.

채점 원칙:
- 표현이 아니라 사실(의미)을 본다. 단어가 달라도 같은 사실을 말하면 담은 것이다. 같은 단어가 있어도 다른 뜻이면 담지 않은 것이다.
- 사실의 일부만 담았으면 담지 않은 것으로 본다(예: 세 요소 중 둘만 말함).
- '말하면 안 되는 것'은 답변이 그 주장을 긍정하거나 사실처럼 제시할 때만 위반이다. 그 주장을 부정·반박하거나 '그렇게 단정할 수 없다'고 말하는 것은 위반이 아니다.
- 답변의 근거 표기([FS-01] 등)는 채점과 무관하다.
- JSON으로만 답한다:
{"facts": [{"fact": "...", "included": true|false, "why": "..."}], "violations": [{"claim": "...", "violated": true|false, "why": "..."}]}"""


def judge_prompt(question: str, answer: str, must_include: list[str], must_not: list[str]) -> tuple[str, str]:
    user = (
        f"질문: {question}\n\n답변:\n{answer}\n\n"
        f"반드시 담아야 할 사실:\n" + "\n".join(f"- {f}" for f in must_include) + "\n\n"
        f"말하면 안 되는 것:\n" + "\n".join(f"- {c}" for c in must_not)
    )
    return JUDGE_SYSTEM, user
