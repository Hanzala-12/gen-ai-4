# Generative Models: MAE, VAE, Diffusion and RAG

Four generative models in one notebook, plus one web app that runs all four.

| Part | Model | Data |
|---|---|---|
| A | Masked autoencoder (small vision transformer) and a few-label linear classifier | EuroSAT RGB |
| B | Variational autoencoder | UHaT Urdu characters |
| C | Class-conditional denoising diffusion (DDPM) | UHaT Urdu characters |
| D | Retrieval-augmented generation for biomedical questions | BioASQ (mini) |

## Repository layout

| Path | Contents |
|---|---|
| `generative_models_assignment.ipynb` | The full implementation. Run it top to bottom on a GPU |
| `generative_models_assignment_executed.ipynb` | A copy executed on Kaggle (Tesla T4) with all outputs |
| `starter/` | The given starter code, unchanged |
| `part_a_mae/`, `part_b_vae/`, `part_c_diffusion/`, `part_d_rag/` | Model code written by the notebook |
| `results/` | Figures, tables, `rag_answers.jsonl`, `human_eval.csv`, `analysis.md`, `samples.md` |
| `app/` | The web front end |
| `weights/` | Trained weights (not committed, see below) |

## Reproduce the results

1. Open `generative_models_assignment.ipynb` on Kaggle or Colab with a GPU and run all cells. It installs its
   own packages and downloads all three datasets. The whole run takes about 36 minutes on a T4.
2. The notebook writes `results/`, `weights/` and the `part_*` code folders.

## Run the web app

```
pip install -r app/requirements.txt
python app/app.py
```

The app needs three things in the repository folder:

- `weights/`: download the trained weights from **WEIGHTS_LINK (add the Hugging Face or Google Drive link here)**
  and unzip them so that `weights/mae_eurosat.pt`, `vae_uhat.pt`, `ddpm_uhat.pt`, `rag_index.pt` and
  `rag_chunks.json` exist.
- `data/uhat/`: the UHaT `data.zip` from <https://zenodo.org/records/3670611>, unzipped there (used by the VAE tab).
- An internet connection the first time, for EuroSAT, MiniLM and Qwen2.5-1.5B-Instruct from Hugging Face.

Tabs: **MAE** (pick a test tile, see masked input, reconstruction and predicted class), **VAE** (random characters
and an interpolation slider between two characters), **Diffusion** (pick a character, generate 8 samples),
**RAG** (ask a question, see the answer, the cited passages with abstract ids, and the medical notice).

The app uses a GPU when one is available. On a CPU the diffusion tab takes around ten minutes for 8 samples
(1000 denoising steps) and the first RAG question loads about 3 GB of models.

Screenshots of the four tabs are in `results/app_*.jpg`: `app_mae.jpg`, `app_vae_samples.jpg`,
`app_vae_interpolation.jpg`, `app_diffusion.jpg` and `app_rag.png` (the RAG tab, captured on a Kaggle GPU).

## Results

| Part | Result |
|---|---|
| A | Few-label classifier test accuracy 46.57 % (chance 10 %), 160 labelled images |
| B | Test negative ELBO 130.05 nats = reconstruction 104.77 + KL 25.28 |
| C | Final L_simple 0.0227 |
| D retrieval | Hit@1 0.622, Hit@5 0.816, Hit@10 0.861, Recall@10 0.602, MRR@10 0.706 |
| D generation (100 questions) | ROUGE-L 0.259, citation precision 0.686, "I don't know" rate 0.060 |

Full tables are in `results/tables.md` and the discussion in `results/analysis.md`.

## Correctness checks

Output of the assignment's checks (also in `results/correctness_checks.md`). All pass.

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

## Human evaluation

`results/human_eval.csv` holds the diffusion blind-test sheet and the 20 RAG answers to review. Both members
fill in their columns by hand; until then the corresponding table cells read "pending".
