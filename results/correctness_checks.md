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
