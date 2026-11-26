"""One web app, four tabs, all running the models trained in the notebook.

Run from the repository root:  python app/app.py
Needs: weights/ (see README), data/uhat/ (UHaT data.zip unzipped) and an internet connection the
first time (EuroSAT, MiniLM and Qwen are downloaded from Hugging Face).
"""
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
UHAT_ROOT = ROOT / "data" / "uhat"
sys.path.insert(0, str(ROOT / "starter"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RETRIEVER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_module(name, rel_path):
    """Several parts have a file called model.py, so load each one under its own name."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mae_mod = load_module("part_a_model", "part_a_mae/model.py")
vae_mod = load_module("part_b_model", "part_b_vae/model.py")
ddpm_mod = load_module("part_c_ddpm", "part_c_diffusion/ddpm.py")
rag_mod = load_module("part_d_rag", "part_d_rag/rag.py")

from eurosat import load_eurosat_split, patchify, unpatchify  # noqa: E402
from uhat import load_uhat_characters  # noqa: E402


def grid(images, nrow, scale=1):
    """images: (N, C, H, W) in [0, 1] -> one uint8 array laid out nrow images per row."""
    images = images.detach().cpu().clamp(0, 1)
    n, c, h, w = images.shape
    rows = (n + nrow - 1) // nrow
    canvas = np.ones((rows * (h + 2), nrow * (w + 2), 3), dtype=np.float32)
    for i in range(n):
        r, col = divmod(i, nrow)
        img = images[i].permute(1, 2, 0).numpy()
        img = np.repeat(img, 3, axis=2) if c == 1 else img
        canvas[r * (h + 2) + 1:r * (h + 2) + 1 + h, col * (w + 2) + 1:col * (w + 2) + 1 + w] = img
    canvas = (canvas * 255).astype(np.uint8)
    if scale > 1:
        canvas = canvas.repeat(scale, axis=0).repeat(scale, axis=1)
    return canvas


# ------------------------------------------------------------------ Part A: MAE
@lru_cache(maxsize=1)
def get_mae():
    ck = torch.load(WEIGHTS / "mae_eurosat.pt", map_location="cpu")
    mae = mae_mod.MAE()
    mae.load_state_dict(ck["mae"])
    probe = mae_mod.LinearProbe()
    probe.load_state_dict(ck["probe"])
    return mae.to(DEVICE).eval(), probe.to(DEVICE).eval(), ck["class_names"]


@lru_cache(maxsize=1)
def get_eurosat_test():
    return load_eurosat_split("test")


def run_mae(index, seed):
    mae, probe, names = get_mae()
    images, labels = get_eurosat_test()
    index = int(index) % len(images)
    x = images[index:index + 1].to(DEVICE)
    torch.manual_seed(int(seed))
    with torch.no_grad():
        patches = patchify(x)
        latent, mask, ids_restore = mae.forward_encoder(patches)
        pred = mae.forward_decoder(latent, ids_restore)
        m = mask.unsqueeze(-1)
        masked_input = patches * (1 - m) + 0.5 * m               # masked patches shown grey
        recon = patches * (1 - m) + pred * m                     # predicted patches pasted in
        probs = torch.softmax(probe(mae.features(patches)), dim=-1)[0].cpu()
    show = lambda p: grid(unpatchify(p).cpu(), 1, scale=4)
    top = {names[i]: float(probs[i]) for i in range(len(names))}
    truth = names[int(labels[index])]
    return show(masked_input), show(recon), show(patches), top, f"True class: **{truth}** (tile {index})"


def random_tile():
    return int(np.random.randint(0, len(get_eurosat_test()[0])))


# ------------------------------------------------------------------ Part B: VAE
@lru_cache(maxsize=1)
def get_vae():
    ck = torch.load(WEIGHTS / "vae_uhat.pt", map_location="cpu")
    vae = vae_mod.VAE()
    vae.load_state_dict(ck["vae"])
    return vae.to(DEVICE).eval(), ck["class_names"]


@lru_cache(maxsize=1)
def get_uhat_test():
    _, names = get_vae()
    x, y, _ = load_uhat_characters(str(UHAT_ROOT), split="test", class_names=names)
    return x, y


def vae_random(n=64):
    vae, _ = get_vae()
    return grid(vae.sample(n, DEVICE), 8, scale=3)


def pick_image(name, seed=0):
    _, names = get_vae()
    x, y = get_uhat_test()
    idx = (y == names.index(name)).nonzero().flatten()
    g = torch.Generator().manual_seed(int(seed))
    return x[idx[torch.randint(len(idx), (1,), generator=g)]].to(DEVICE)


def vae_interpolate(name_a, name_b, w):
    vae, _ = get_vae()
    a, b = pick_image(name_a), pick_image(name_b)
    with torch.no_grad():
        mu_a, _ = vae.encode(a)
        mu_b, _ = vae.encode(b)
        one = torch.sigmoid(vae.decode((1 - w) * mu_a + w * mu_b))
        strip = vae.interpolate(a, b, steps=10)
    return grid(torch.cat([a, b]), 2, scale=4), grid(one, 1, scale=6), grid(strip, 10, scale=3)


# ------------------------------------------------------------------ Part C: diffusion
@lru_cache(maxsize=1)
def get_ddpm():
    ck = torch.load(WEIGHTS / "ddpm_uhat.pt", map_location="cpu")
    unet = ddpm_mod.build_unet()
    unet.load_state_dict(ck["unet_ema"])
    return unet.to(DEVICE).eval(), ddpm_mod.LinearNoiseSchedule(device=DEVICE), ck["class_names"]


def generate_character(name, seed):
    unet, sched, names = get_ddpm()
    labels = torch.full((8,), names.index(name), device=DEVICE, dtype=torch.long)
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    x, _ = ddpm_mod.ddpm_sample(unet, sched, labels, generator=g)
    return grid((x + 1) / 2, 4, scale=4)


# ------------------------------------------------------------------ Part D: RAG
@lru_cache(maxsize=1)
def get_rag():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    chunks = json.load(open(WEIGHTS / "rag_chunks.json", encoding="utf8"))
    idx = torch.load(WEIGHTS / "rag_index.pt", map_location="cpu")
    index = rag_mod.DenseIndex(idx["embeddings"], idx["pids"], DEVICE)
    encoder = rag_mod.MeanPoolEncoder(RETRIEVER_NAME, DEVICE, max_length=256)
    tok = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    gen = AutoModelForCausalLM.from_pretrained(GENERATOR_NAME, dtype=dtype).to(DEVICE).eval()
    return chunks, index, encoder, tok, gen


def answer_question(question):
    if not question.strip():
        return "Type a question first.", ""
    chunks, index, encoder, tok, gen = get_rag()
    q = encoder.encode([question])
    ids, _ = index.search_chunks(q, k=5)
    passages = [chunks["texts"][i] for i in ids[0]]
    abstract_ids = [chunks["pids"][i] for i in ids[0]]
    text = tok.apply_chat_template(rag_mod.build_messages(question, passages), tokenize=False,
                                   add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = gen.generate(**enc, max_new_tokens=256, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    cited = rag_mod.parse_citations(answer)
    sources = "\n\n".join(
        f"**[{i}] abstract {pid}**{' (cited)' if i in cited else ''}\n\n{p}"
        for i, (pid, p) in enumerate(zip(abstract_ids, passages), start=1))
    return f"{answer}\n\n*{rag_mod.MEDICAL_NOTICE}*", sources


# ------------------------------------------------------------------ interface
def build_ui():
    names = torch.load(WEIGHTS / "vae_uhat.pt", map_location="cpu")["class_names"]
    with gr.Blocks(title="Four generative models") as demo:
        gr.Markdown("# Masked autoencoder, VAE, diffusion and RAG\n"
                    f"All four tabs run the models trained in the notebook (device: `{DEVICE}`).")
        with gr.Tabs():
            with gr.Tab("MAE: satellite tiles"):
                gr.Markdown("Pick a EuroSAT test tile. 75% of its patches are hidden; the MAE fills them "
                            "in, and a linear classifier trained on 160 labelled images predicts the land cover.")
                with gr.Row():
                    idx = gr.Number(value=0, label="Test tile index (0-5399)", precision=0)
                    seed = gr.Number(value=0, label="Mask seed", precision=0)
                    rnd = gr.Button("Random tile")
                    go = gr.Button("Run MAE", variant="primary")
                truth = gr.Markdown()
                with gr.Row():
                    a = gr.Image(label="Masked input", height=300)
                    b = gr.Image(label="Reconstruction", height=300)
                    c = gr.Image(label="Original", height=300)
                lab = gr.Label(label="Predicted land cover", num_top_classes=5)
                rnd.click(random_tile, outputs=idx)
                go.click(run_mae, [idx, seed], [a, b, c, lab, truth])
            with gr.Tab("VAE: handwriting"):
                with gr.Row():
                    new = gr.Button("New random characters", variant="primary")
                samples = gr.Image(label="64 characters decoded from z ~ N(0, I)", height=420)
                new.click(vae_random, outputs=samples)
                gr.Markdown("### Interpolate between two characters")
                with gr.Row():
                    na = gr.Dropdown(names, value=names[0], label="Character A")
                    nb = gr.Dropdown(names, value=names[1], label="Character B")
                    w = gr.Slider(0, 1, value=0.5, step=0.05, label="Position between A and B")
                with gr.Row():
                    ends = gr.Image(label="A (left) and B (right)", height=180)
                    one = gr.Image(label="Decoded at the slider position", height=180)
                strip = gr.Image(label="Ten equally spaced points", height=120)
                for ctl in (na, nb, w):
                    ctl.change(vae_interpolate, [na, nb, w], [ends, one, strip])
                demo.load(vae_interpolate, [na, nb, w], [ends, one, strip])
            with gr.Tab("Diffusion: write a character"):
                gr.Markdown("Choose a character; the DDPM denoises pure noise into eight new handwritten "
                            "versions (1000 denoising steps: a few seconds on a GPU, around ten minutes on a CPU).")
                with gr.Row():
                    ch = gr.Dropdown(names, value=names[0], label="Character")
                    dseed = gr.Number(value=0, label="Seed", precision=0)
                    dgo = gr.Button("Generate 8 samples", variant="primary")
                out = gr.Image(label="Eight samples", height=300)
                dgo.click(generate_character, [ch, dseed], out)
            with gr.Tab("RAG: biomedical questions"):
                gr.Markdown("Ask a research question. The top 5 passages from 40k PubMed abstracts are "
                            "retrieved and the answer cites them. The first question loads the models "
                            "(a few minutes on CPU).")
                q = gr.Textbox(label="Question", placeholder="Is capmatinib effective for glioblastoma?")
                ask = gr.Button("Answer", variant="primary")
                ans = gr.Markdown(label="Answer")
                src = gr.Markdown(label="Cited passages")
                ask.click(answer_question, q, [ans, src])
    return demo


if __name__ == "__main__":
    torch.seed()   # PyTorch's default seed is fixed; reseed so every launch gives new random samples
    build_ui().launch()
