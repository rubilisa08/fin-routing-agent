| 시각 | 라벨 | 모델 | 채점기 | 설정 | 도구 호출 | 답변 | ① 판정 |
|---|---|---|---|---|---|---|---|
| 2026-11-02 12:42 | offline-baseline | offline/rule-based | keyword | top_k=4, scope=0, term=0, gate=1, sim=0.05 | 16/20 (80%) | 8/20 (40%) | 20/20 |
| 2026-11-02 12:43 | offline+term-coverage | offline/rule-based | keyword | top_k=4, scope=0, term=1, gate=1, sim=0.05 | 18/20 (90%) | 10/20 (50%) | 20/20 |
| 2026-11-02 12:44 | offline+scope-rules | offline/rule-based | keyword | top_k=4, scope=1, term=1, gate=1, sim=0.05 | 20/20 (100%) | 12/20 (60%) | 20/20 |
| 2026-11-02 12:47 | offline+top_k6 | offline/rule-based | keyword | top_k=6, scope=1, term=1, gate=1, sim=0.05 | 20/20 (100%) | 12/20 (60%) | 20/20 |
| 2026-11-02 12:53 | offline+answer-filter | offline/rule-based | keyword | top_k=4, scope=1, term=1, gate=1, sim=0.05 | 20/20 (100%) | 12/20 (60%) | 20/20 |
| 2026-11-02 12:59 | offline-holdout | offline/rule-based | keyword | top_k=4, scope=1, term=1, gate=1, sim=0.05 | 6/10 (60%) | 7/10 (70%) | 7/10 |
