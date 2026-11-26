"""Chunking, MiniLM encoder with mean pooling, dense index, retrieval metrics and prompting."""
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MEDICAL_NOTICE = "For educational use only. Not medical advice."
IDK_SENTENCE = "I don't know based on the provided passages."


# ---------------------------------------------------------------- chunking
def chunk_words(text, size=150, overlap=30):
    """Windows of at most `size` words; consecutive windows share `overlap` words."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def build_chunks(corpus, size=150, overlap=30):
    texts, pids = [], []
    for pid, text in corpus.items():
        for c in chunk_words(text, size, overlap):
            texts.append(c)
            pids.append(pid)
    return texts, pids


# ---------------------------------------------------------------- encoder
def mean_pool(last_hidden, attention_mask):
    """Average of the token states over real tokens only (padding excluded)."""
    m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


class MeanPoolEncoder:
    def __init__(self, name, device, max_length=256):
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        if device.type == "cuda":
            self.model.half()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, batch_size=256):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length bucketing
        out = torch.empty(len(texts), self.dim)
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state.float()
            out[idx] = F.normalize(mean_pool(h, enc["attention_mask"].float()), dim=-1).cpu()
        return out

    def token_lengths(self, texts, batch_size=1024):
        lengths = []
        for s in range(0, len(texts), batch_size):
            lengths += [len(ids) for ids in self.tok(texts[s:s + batch_size])["input_ids"]]
        return lengths


# ---------------------------------------------------------------- index and search
class DenseIndex:
    def __init__(self, embeddings, chunk_pids, device):
        self.device = device
        self.emb = embeddings.to(device)
        self.chunk_pids = list(chunk_pids)
        self.abstract_ids = sorted(set(self.chunk_pids))
        col = {pid: j for j, pid in enumerate(self.abstract_ids)}
        self.chunk_col = torch.tensor([col[p] for p in self.chunk_pids], device=device)

    def chunk_scores(self, q):
        """Cosine similarity of every question with every chunk: one matrix multiplication."""
        return q.to(self.device, self.emb.dtype) @ self.emb.T

    @torch.no_grad()
    def search_abstracts(self, q, k=10, batch_size=128):
        """An abstract's score is the maximum score of its chunks. Returns top-k ids and scores."""
        ids, scores = [], []
        for s in range(0, len(q), batch_size):
            cs = self.chunk_scores(q[s:s + batch_size]).float()
            ab = torch.full((len(cs), len(self.abstract_ids)), float("-inf"), device=self.device)
            ab.scatter_reduce_(1, self.chunk_col.unsqueeze(0).expand(len(cs), -1), cs,
                               reduce="amax", include_self=True)
            top = ab.topk(min(k, ab.shape[1]), dim=1)
            ids += [[self.abstract_ids[j] for j in row] for row in top.indices.tolist()]
            scores += top.values.tolist()
        return ids, scores

    @torch.no_grad()
    def search_chunks(self, q, k=5):
        top = self.chunk_scores(q).float().topk(k, dim=1)
        return top.indices.tolist(), top.values.tolist()


# ---------------------------------------------------------------- metrics
def hit_at_k(ranked, relevant, k):
    return float(len(set(ranked[:k]) & set(relevant)) >= 1)


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / min(k, len(set(relevant)))


def mrr_at_k(ranked, relevant, k=10):
    rel = set(relevant)
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(rankings, relevants, ks=(1, 5, 10)):
    res = {}
    for k in ks:
        res[f"Hit@{k}"] = sum(hit_at_k(r, g, k) for r, g in zip(rankings, relevants)) / len(rankings)
        res[f"Recall@{k}"] = sum(recall_at_k(r, g, k) for r, g in zip(rankings, relevants)) / len(rankings)
    res["MRR@10"] = sum(mrr_at_k(r, g, 10) for r, g in zip(rankings, relevants)) / len(rankings)
    return res


# ---------------------------------------------------------------- prompting
SYSTEM_PROMPT = (
    "You are an assistant that answers biomedical research questions for educational purposes. "
    "Answer using only the information in the numbered passages provided by the user. "
    "Cite the passages that support each statement with their numbers in square brackets, for example [2] or [1][3]. "
    "Do not use outside knowledge. If the passages do not contain the answer, reply exactly: "
    + IDK_SENTENCE)


def build_messages(question, passages):
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    user = f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer concisely, citing passage numbers."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_CITE = re.compile(r"\[(\s*\d+\s*(?:[,;\-–]\s*\d+\s*)*)\]")


def parse_citations(text, n_passages=5):
    """Set of passage numbers cited as [2], [1, 3], [1-3] or [1][4]."""
    cited = set()
    for group in _CITE.findall(text):
        for part in re.split(r"[,;]", group):
            part = part.strip()
            if re.fullmatch(r"\d+\s*[\-–]\s*\d+", part):
                a, b = map(int, re.split(r"[\-–]", part))
                cited.update(range(min(a, b), max(a, b) + 1))
            elif part.isdigit():
                cited.add(int(part))
    return {c for c in cited if 1 <= c <= n_passages}


_IDK = re.compile(r"i do(?:n['’]t| not) know|(?:passages|context|information provided) "
                  r"(?:do(?:es)? not|don['’]t) (?:contain|provide|mention|include)", re.I)


def is_idk(text):
    return bool(_IDK.search(text))
