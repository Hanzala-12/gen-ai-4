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
`app_vae_interpolation.jpg`, `app_diffusion.jpg` and `app_rag.jpg`.

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

The output of the assignment's correctness checks is in `results/correctness_checks.md`. All of them pass:
48 of 64 patches masked and loss on masked patches only (MAE), KL never negative and 0 at mu = 0, log sigma^2 = 0 (VAE),
alpha_bar_T of about 4e-5 (DDPM), chunks of at most 150 words, unit-norm embeddings and a perfect retriever scoring 1 (RAG).

## Human evaluation

`results/human_eval.csv` holds the diffusion blind-test sheet and the 20 RAG answers to review. Both members
fill in their columns by hand; until then the corresponding table cells read "pending".
