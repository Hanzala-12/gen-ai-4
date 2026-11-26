# Generative Models Assignment

Masked autoencoder, VAE, denoising diffusion and retrieval-augmented generation, implemented in one notebook.

- `generative_models_assignment.ipynb` is the source notebook (no outputs).
- `generative_models_assignment_executed.ipynb` is the same notebook executed end to end on Kaggle (Tesla T4, 34 min, no errors) with all outputs.

## Results from the Kaggle run

| Part | Model | Result |
|---|---|---|
| A | Masked autoencoder, EuroSAT (3.2M params, 100 epochs pretraining, 160 labels) | Linear-probe test accuracy 46.57 % (chance 10 %); final val masked MSE 0.00687 |
| B | VAE, UHaT Urdu characters (220k params, 30 epochs) | Test negative ELBO 130.12 nats (reconstruction 104.76 + KL 25.35) |
| C | Class-conditional DDPM (4.0M-param U-Net, 40 epochs) | Final L_simple 0.0227; 320 samples (40 characters x 8) in 124 s |
| D | RAG over 40,181 PubMed abstracts, Qwen2.5-1.5B-Instruct | Hit@1 0.622, Hit@10 0.861, MRR@10 0.706; ROUGE-L 0.259, citation precision 0.686 |

Notes:

- Part A confuses visually similar classes most (AnnualCrop and PermanentCrop, 248 mistakes between them).
- Part D failed to retrieve any relevant abstract for 98 of 707 questions (13.9 %). 18.3 % of chunks exceed the encoder's 256-token limit and are truncated.
- The human evaluation sections (diffusion blind test, RAG review) are still pending: the member columns in `human_eval.csv` were not filled in.
