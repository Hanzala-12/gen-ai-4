"""Small tool for the human evaluation. Each member runs it once and saves their own sheet.

Run from the repository root:  python app/label_tool.py
Writes results/human_labels_<name>.csv (columns: task,item,answer,supported,correct).
"""
import csv
import re
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
NAMES = ["alif", "alif mad aa", "ayn", "baa", "bari yaa", "cheey", "choti yaa", "daal", "dhaal", "faa", "gaaf",
         "ghain", "haa1", "haa2", "haa3", "hamza", "jeem", "kaaf", "khaa", "laam", "meem", "noon", "noonghunna",
         "paa", "qaaf", "raa", "rhraa", "seen", "seey", "sheen", "swaad", "taa", "ttaa", "twa", "waw", "zaaa",
         "zaal", "zhaa", "zwaa", "zwaad"]

review = (RESULTS / "rag_human_review.md").read_text(encoding="utf8")
parts = re.split(r"(?m)^## Item (\d+):", review)
ITEMS = [(parts[i], "## Item " + parts[i] + ":" + parts[i + 1]) for i in range(1, len(parts), 2)]


def save(name, *vals):
    name = re.sub(r"\W+", "", name or "") or "member"
    dd, rr = vals[:40], vals[40:]
    rows = [["task", "item", "answer", "supported", "correct"]]
    rows += [["diffusion", i + 1, dd[i] or "", "", ""] for i in range(40)]
    for k, (item, _) in enumerate(ITEMS):
        s, c = rr[2 * k], rr[2 * k + 1]
        rows.append(["rag", item, "", "" if s is None else int(s), "" if c is None else int(c)])
    out = RESULTS / f"human_labels_{name}.csv"
    with open(out, "w", newline="", encoding="utf8") as f:
        csv.writer(f).writerows(rows)
    missing = sum(1 for r in rows[1:] if (r[0] == "diffusion" and not r[2]) or (r[0] == "rag" and (r[3] == "" or r[4] == "")))
    return f"Saved {out} ({missing} answers still empty)."


with gr.Blocks(title="Human evaluation") as demo:
    gr.Markdown("# Human evaluation\nWork alone, without looking at anyone else's answers. Fill both tabs, then press Save.")
    who = gr.Textbox(label="Your name (used in the file name)")
    inputs = []
    with gr.Tab("Diffusion blind test"):
        gr.Markdown("Pick the character each numbered image shows (numbers run left to right, top to bottom, 8 per row).")
        gr.Image(str(RESULTS / "diffusion_blind_test.png"), show_label=False, interactive=False)
        for i in range(40):
            inputs.append(gr.Dropdown(NAMES, label=f"Image {i + 1}"))
    with gr.Tab("RAG answers"):
        for item, text in ITEMS:
            gr.Markdown(text)
            with gr.Row():
                inputs.append(gr.Radio([("yes", 1), ("no", 0)], label=f"Item {item}: supported (everything it says is in the passages)"))
                inputs.append(gr.Radio([("yes", 1), ("no", 0)], label=f"Item {item}: correct (agrees with the reference answer)"))
    btn = gr.Button("Save", variant="primary")
    msg = gr.Markdown()
    btn.click(save, [who] + inputs, msg)

if __name__ == "__main__":
    demo.launch()
