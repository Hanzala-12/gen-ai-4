# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

The whole project is a single notebook, `generative_models_assignment.ipynb`. There is no build system, package manifest, linter or test suite. "Running the project" means running the notebook top to bottom on a CUDA GPU (written for Colab or Kaggle T4, mixed precision when a GPU is present). It installs its own dependencies in the first code cell (`%pip install -q torch transformers datasets diffusers rouge-score matplotlib pillow`) and downloads its own data (EuroSAT, UHaT Urdu characters from Zenodo, BioASQ/PubMed).

On Kaggle, use "Save & Run All (Commit)" with a GPU accelerator to run unattended; the interactive session stops when the browser closes.

## Notebook architecture

Sections run in order and later parts reuse shared code from section 2, so cells are not independent:

- **2.x Setup**: package install, starter files written to `starter/` verbatim from the assignment appendix (do not modify them), shared imports and utilities, and a run configuration cell that controls training sizes and epochs.
- **Task 0**: runs `check_starter.py` and reports shapes and example grids.
- **Part A**: masked autoencoder on EuroSAT RGB, pretrained without labels, then a few-label linear classifier, accuracy and confusion matrix.
- **Parts B and C**: VAE and class-conditional denoising diffusion (DDPM) on UHaT Urdu characters.
- **Part D**: retrieval-augmented generation over PubMed abstracts (chunking, embedding index, search, generation, ROUGE-style evaluation).
- **Section 8**: human evaluation.

## Constraints that shape any edit

- Library limits are an assignment rule, not a preference. Allowed building blocks are only `nn.TransformerEncoderLayer` / `nn.TransformerEncoder` (Part A), `diffusers.UNet2DModel` (Part C, architecture only), and `AutoTokenizer`, `AutoModel`, `AutoModelForCausalLM`, `generate` and `apply_chat_template` from `transformers` (Part D only).
- No pretrained weights in Parts A to C. No diffusers scheduler or pipeline, `timm`, `sentence-transformers`, vector database or `scikit-learn`. The noise schedule, diffusion objective and sampler, masking, ELBO, vector index, nearest-neighbour search, confusion matrix and retrieval metrics are hand-written in PyTorch.
- Each part has a "Correctness checks" cell; keep these passing when changing models or losses.
