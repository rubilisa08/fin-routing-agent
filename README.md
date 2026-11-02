# AI Financial Learning & Analysis Agent

재무·회계 질문을 **분야별로 분류**하고, 그 분야의 **학습노트에서만 근거를 찾아 답하며**, 근거가 없으면 **지어내지 않고 넘기는** 라우팅 에이전트.

```
입력 → ① 카테고리 판정 → ② 카테고리별 근거 조립(모르면 넘김) → ③ 근거만으로 답변 → ④ 검증
```

설계 판단과 측정 결과는 [REPORT.md](REPORT.md)에 있다.

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY를 채운다. 비워 두면 offline 모드
```

```bash
streamlit run app.py                         # 데모 화면
python agent.py "듀폰분석이 뭐야?"              # 질문 하나
python evaluate.py --validate-grader         # 채점기 자체 검증 (모범/오답 답안)
python evaluate.py --label baseline          # 평가셋 20문항 측정 → results/
python evaluate.py --data data/holdout.json --label holdout   # 미공개 문항으로 과적합 점검
```

손잡이 하나만 바꿔 다시 측정: `--top-k 6`, `--no-scope-rules`, `--no-term-coverage`, `--no-llm-gate`, `--provider offline`.

## 구조

```
docs/                 근거 문서 4개 (섹션마다 [FS-01] 같은 ID)
data/goldenset.json   평가셋 20문항 (기대 도구 · 필수 사실 · 금지 표현 · 모범/오답 답안)
data/holdout.json     과적합 점검용 미공개 10문항
data/fewshot.json     프롬프트 예시 (평가셋과 분리)
prompts.py            분류 지침 · 근거 충분성 판단 · 답변 규칙 · 채점 지침
context.py            문서 쪼개기 · 카테고리↔섹션 매핑표 · 근거 조립과 넘기기 판단
agent.py              LangGraph 파이프라인
verify.py             기계적 검증 (근거 표기·인용 범위·수치·용어·투자 권유)
llm.py                Gemini / OpenAI / offline 백엔드
evaluate.py           도구 호출 적절성 · 답변 적절성 측정, 채점기 검증, 오답 분석
app.py                Streamlit 데모
results/              측정 기록 (history.md = 개선 시도별 기록표)
```

> 근거 문서는 기준서·법령을 요약한 학습노트이며, 실무 판단이나 투자 조언이 아니다.
