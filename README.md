# FinRoute AI

**Financial & Accounting Knowledge Routing Agent.** 재무·회계 질문을 4개 분야(회계원리·재무제표·재무분석·기업재무)로 **분류**하고, 그 분야의 **근거 문서에서만 답하며**, 분야가 모호하면 **되묻고**, 근거가 없으면 **지어내지 않고 보류**한다.

```
입력 → ① 판정(신뢰도) → 신뢰도 점검(낮으면 되묻기) → ② 근거 조립(부족하면 보류) → ③ 근거만으로 답변 → ④ 검증
```

설계 판단과 측정 결과는 [REPORT.md](REPORT.md)에 있다.

## 실행 (Windows PowerShell 기준)

```powershell
pip install -r requirements.txt
```

`.env.example`을 복사해 이름을 `.env`로 바꾸고 API 키를 넣는다. 키가 없으면 offline 모드로 동작한다.

```powershell
python -m streamlit run app.py
```

```powershell
python agent.py "듀폰분석이 뭐야?"
```

```powershell
python evaluate.py --validate-grader
```

```powershell
python evaluate.py --label baseline
```

```powershell
python evaluate.py --data data/holdout.json --label holdout
```

손잡이 하나만 바꿔 다시 측정: `--conf 0.7`(신뢰도 임계값), `--top-k 6`, `--no-scope-rules`, `--no-term-coverage`, `--no-llm-gate`, `--provider offline`, `--grader keyword`.

LLM 응답은 `.llm_cache/`에 저장되어 같은 호출은 다시 하지 않는다. 무료 호출 한도에 걸려 평가가 멈추면 같은 명령을 다시 실행해 이어서 진행한다.

## 구조

```
docs/                 근거 문서 6개 · 52개 섹션 (섹션마다 ID·출처·버전)
data/goldenset.json   평가셋 50문항 (기대 도구 · 필수 사실 · 금지 표현 · 모범/오답 답안)
data/holdout.json     과적합 점검용 미공개 10문항
data/fewshot.json     프롬프트 예시 (평가셋과 분리)
prompts.py            분류 지침 · 근거 충분성 판단 · 답변 규칙 · 채점 지침
context.py            문서 쪼개기 · 카테고리↔섹션 매핑표 · 근거 조립과 보류 판단
agent.py              LangGraph 파이프라인
verify.py             기계적 검증 (출처·인용 범위·카테고리 일치·계산 재검산·수치·용어·투자 권유)
llm.py                Gemini / OpenAI / offline 백엔드, 응답 캐시
evaluate.py           과제 지표 2개 + PRD 지표 5개, 채점기 검증, 오답 분석
app.py                Streamlit 데모
results/              측정 기록 (history.md = 개선 시도별 기록표, v1/ = 1차 버전 기록)
```

> 근거 문서는 기준서·법령·교재를 요약한 학습노트이며, 실무 판단이나 투자 조언이 아니다.
