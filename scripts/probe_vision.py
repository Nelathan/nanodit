"""PLAN I1 (priming, topics/priming.md): what input size EmbeddingGemma 2's vision tower needs on our images, on CPU.

The first N eval latents (256 px) are decoded by Primal and embedded as images, the processor resizing each to a budget
of `max_soft_tokens`, each a 3x3 pool of 16 px patches; it takes only 70, 140, 280 (the default), 560 and 1120. A square
image goes to 384, 528 and 768 px at the first three: at 70 its soft tokens are our 8x8 grid at 128 px, at 280 the
16x16 grid at 256 px.
At each budget: the cosine of each image to its own caption against the others (the stored pooled caption, which carries the document prefix,
and the caption under the retrieval query prefix), and top-1 retrieval of the caption among the N.
Usage: probe_vision.py [N]
"""

import json
import sys
import time

import pyarrow.parquet as pq
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from nanodit.data import _bf16
from nanodit.model import LATENT_CHANNELS, TEXT_DIM
from train import load_decoder

EVAL = "/run/media/djg/vel-ssd/tiny-dit/lunara_eval_256_seq512.parquet"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.set_grad_enabled(False)

table = pq.read_table(EVAL, columns=["caption", "embedding", "latent_256"]).slice(0, N)
captions = table["caption"].to_pylist()
stored = _bf16(table["embedding"], TEXT_DIM).float()
latents = _bf16(table["latent_256"], LATENT_CHANNELS, 128, 128).float()

decoder = load_decoder(torch.device("cpu"))
images = torch.cat([decoder.decode(latents[i : i + 16]).clamp(-1, 1) for i in range(0, N, 16)])
pils = [Image.fromarray(((im + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()) for im in images]

st = SentenceTransformer("google/embeddinggemma-2", device="cpu", config_kwargs={"audio_config": None}, model_kwargs={"dtype": torch.float32})
query = st.encode(captions, prompt_name="query", convert_to_tensor=True, normalize_embeddings=True)
stored = torch.nn.functional.normalize(stored, dim=1)


def read(sim: torch.Tensor) -> dict:
    own = sim.diag()
    other = (sim.sum(1) - own) / (N - 1)
    return {"own": own.mean().item(), "other": other.mean().item(), "top1": (sim.argmax(1) == torch.arange(N)).float().mean().item()}


processor = st[0].processor.image_processor
for budget in (70, 140, 280):
    processor.max_soft_tokens = budget
    start = time.time()
    emb = st.encode(pils, convert_to_tensor=True, normalize_embeddings=True, batch_size=16)
    seconds = time.time() - start
    print(json.dumps({"max_soft_tokens": budget, "img_per_s": N / seconds, "stored_caption": read(emb @ stored.T), "query_caption": read(emb @ query.T)}), flush=True)
