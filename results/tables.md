# Results

**Table 1 – Data**

| Dataset | Quantity |
|---|---|
| EuroSAT | train / test images: 16,200 / 5,400; labelled images used: 160 |
| UHaT | character classes: 40; train / test images: 28,328 / 4,747 |
| BioASQ | abstracts: 40,181; chunks: 79,332; test questions: 707 |
| BioASQ | mean number of relevant abstracts per test question: 8.66 |

**Table 2 – Training**

| Model | Parameters | Final training loss | Time / GPU |
|---|---|---|---|
| MAE (encoder + decoder) | 3,173,248 | 0.0067 (masked MSE) | 3.7 min / Tesla T4 |
| VAE | 220,385 | 129.97 (negative ELBO, nats) | 0.4 min / Tesla T4 |
| DDPM U-Net | 4,028,353 | 0.0227 (L_simple) | 21.5 min / Tesla T4 |

**Table 3 – Image results**

| Result | Value |
|---|---|
| MAE few-label classifier: test accuracy | 46.57 % |
| VAE: test negative ELBO = reconstruction + KL (nats) | 130.05 = 104.77 + 25.28 |
| Diffusion: blind test, member 1 / member 2 correct (of 40) | pending |
| Diffusion: images where both members agreed (of 40) | pending |

**Table 4 – Retrieval**

| Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| 0.622 | 0.816 | 0.861 | 0.622 | 0.587 | 0.602 | 0.706 |

**Table 5 – Generation (100 questions)**

| Metric | Value |
|---|---|
| ROUGE-L | 0.259 |
| Citation precision | 0.686 |
| "I don't know" rate | 0.060 |
| Human: supported, member 1 / member 2 (of 20) | pending |
| Human: correct, member 1 / member 2 (of 20) | pending |

## Correctness checks

**MAE**
- masked patches per image: min 48, max 48 (expected 48 of 64)
- distinct masks among 256 images: 256; identical masks between two calls: 0 of 256
- loss before / after corrupting visible-patch predictions: 1.104597 / 1.104597
- max |grad| at visible patches = 0.0e+00, at masked patches = 3.0e-06

**VAE**
- KL at mu = 0, log sigma^2 = 0: [-0.0, -0.0, -0.0, -0.0]
- KL on 100,000 random (mu, log sigma^2): min 29.3455 (must be >= 0)
- encoder output (8, 16), decoder output (8, 1, 28, 28)
- smallest per-image KL over all training batches: 0.0069 (never negative)

**DDPM**
- alpha_bar_T = 4.036e-05 (expected about 4e-5); sqrt(alpha_bar_T) = 0.0064
- x_999 over 512 images: mean -0.0064, std 0.9990 (N(0,1) expected); correlation with x_0: 0.0048
- U-Net output shape (4, 1, 28, 28) for input (4, 1, 28, 28)

**RAG**
- largest chunk: 150 words (limit 150)
- embedding norms (chunks and questions): min 1.000000, max 1.000000
- padding invariance of mean pooling: cosine(alone, padded batch) = 1.00000
- perfect retriever: Hit@1 = 1.000, Recall@1 = 1.000, Hit@5 = 1.000, Recall@5 = 1.000, Hit@10 = 1.000, Recall@10 = 1.000, MRR@10 = 1.000
