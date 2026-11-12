"""Archive 2026-11-12-embedding-norm: EmbeddingGemma 2 text path via sentence-transformers: output norm, per-dim statistics, bf16 vs fp32, short prompts, on CPU.

The raw (pre-Normalize) vectors come from the Transformer + Pooling modules; L2 normalization is the third module.

usage: probe_embeddinggemma.py SHARD.tar OUT_DIR
"""
import json
import sys
import tarfile
import time

import torch
from sentence_transformers import SentenceTransformer

OUT = sys.argv[2]
REPO = "google/embeddinggemma-2"
PREFIX = "title: none | text: "
N_CAPTIONS = 256
CONTEXT = 8192  # model card; config.json text_config.max_position_embeddings says 262144
SHORT = [
    "a red apple on a table",
    "a cat sleeping on a sofa",
    "a mountain lake at sunrise",
    "a portrait of an old man with a beard",
    "a city street at night in the rain",
    "a bowl of ramen",
    "a blue sports car on a highway",
    "two children playing in the snow",
    "a vase of yellow flowers on a windowsill",
    "a robot holding an umbrella",
    "a wooden boat on a calm sea",
    "a black and white dog running on a beach",
    "a slice of chocolate cake on a white plate",
    "a futuristic city skyline",
    "a woman in a red dress",
    "a bicycle leaning against a brick wall",
]

captions = []
with tarfile.open(sys.argv[1], "r|") as tar:
    for m in tar:
        if m.name.endswith(".json"):
            captions.append(json.loads(tar.extractfile(m).read())["prompt"])
            if len(captions) == N_CAPTIONS:
                break

runs = {}
for name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
    st = SentenceTransformer(REPO, device="cpu", config_kwargs=dict(vision_config=None, audio_config=None), model_kwargs={"dtype": dtype})
    if name == "fp32":
        tok = st.tokenizer
        ids = [tok(PREFIX + c)["input_ids"] for c in captions + SHORT]
        lens = torch.tensor([len(i) for i in ids])
        n_params = sum(p.numel() for p in st.parameters())
        print("params", n_params)
    t0 = time.time()
    raw = SentenceTransformer(modules=[st[0], st[1]]).encode(captions + SHORT, prompt=PREFIX, batch_size=16, convert_to_tensor=True).float()  # Transformer + Pooling, before Normalize
    runs[name] = (raw, time.time() - t0)
    del st


def q(x):  # min / median / max
    return {"min": x.min().item(), "median": x.median().item(), "max": x.max().item()}


res = {
    "n_captions": N_CAPTIONS, "n_short": len(SHORT), "params": n_params,
    "token_len_captions": {**q(lens[:N_CAPTIONS].float()), "mean": lens[:N_CAPTIONS].float().mean().item()},
    "token_len_short": q(lens[N_CAPTIONS:].float()),
}
res["context_tokens"] = CONTEXT
res["n_over_context"] = int((lens > CONTEXT).sum())

unit = {}
for name, (raw, secs) in runs.items():
    e = torch.nn.functional.normalize(raw, dim=-1)
    unit[name] = e
    cap = e[:N_CAPTIONS]
    scaled = cap * 768**0.5
    res[name] = {
        "output_dim": raw.shape[1],
        "raw_pooled_norm": q(raw.norm(dim=-1)),
        "normalized_norm": q(e.norm(dim=-1)),
        "normalized_then_rounded_to_bf16_norm": q(e.bfloat16().float().norm(dim=-1)),
        "nan_or_inf": bool((~torch.isfinite(raw)).any()),
        "per_dim_mean_abs_over_captions": q(cap.mean(0).abs() * 768**0.5),  # of the sqrt(768)-scaled embedding
        "per_dim_std_over_captions_scaled": q(scaled.std(0)),
        "scaled_rms_over_all_entries": scaled.pow(2).mean().sqrt().item(),
        "scaled_std_over_all_entries": scaled.std().item(),
        "scaled_per_caption_rms": q(scaled.pow(2).mean(1).sqrt()),
        "captions_per_s": (len(captions) + len(SHORT)) / secs,
    }

cos = (unit["bf16"] * unit["fp32"]).sum(-1)
res["bf16_vs_fp32_cosine"] = {**q(cos), "mean": cos.mean().item()}

e = unit["fp32"]
cap, short = e[:N_CAPTIONS], e[N_CAPTIONS:]
cc = cap @ cap.T
off = cc[~torch.eye(N_CAPTIONS, dtype=torch.bool)]
cc.fill_diagonal_(-1)
ss = short @ cap.T
sh = short @ short.T
sh = sh[~torch.eye(len(SHORT), dtype=torch.bool)]
nearest = ss.max(1)
res["caption_caption_cosine"] = {"all_pairs": {**q(off), "mean": off.mean().item()}, "each_to_its_nearest_other_caption": {**q(cc.max(1).values), "mean": cc.max(1).values.mean().item()}}
res["short_to_captions_cosine"] = {"nearest": {**q(nearest.values), "mean": nearest.values.mean().item()}, "all_pairs_mean": ss.mean().item(), "all_pairs_median": ss.median().item()}
res["short_short_cosine"] = {**q(sh), "mean": sh.mean().item()}
res["short_nearest_pairs"] = [{"prompt": SHORT[i], "cos": nearest.values[i].item(), "caption": captions[nearest.indices[i]][:140]} for i in range(len(SHORT))]
print(json.dumps(res, indent=1))
json.dump(res, open(f"{OUT}/embeddinggemma.json", "w"), indent=1)
