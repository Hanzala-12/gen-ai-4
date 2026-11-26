# AI-judge experiment (not the human evaluation)

Two AI judges (separate model instances, no access to the answer key or to each other's output) did the same two tasks that the assignment gives to the two human members. Their raw answers are in `ai_judge_1.csv` and `ai_judge_2.csv`. These are **not** the human evaluation: the member 1 / member 2 columns of `human_eval.csv` and the "pending" cells of Tables 3 and 5 are left for the human members.

| Task | AI judge 1 | AI judge 2 |
|---|---|---|
| Diffusion blind test, correct (of 40) | 15 | 17 |
| Diffusion, both judges correct (of 40) | 10 | |
| Diffusion, both judges gave the same label (of 40) | 19 | |
| RAG answers supported (of 20) | 15 | 15 |
| RAG answers correct (of 20) | 11 | 11 |

Reading: roughly 40 % of generated characters were recognised, far above the 2.5 % of random guessing, with the dot-differing letter families as the likely confusions. About three quarters of the sampled RAG answers were fully supported by their passages and just over half agreed with the reference.
