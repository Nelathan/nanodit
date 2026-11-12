"""Which EmbeddingGemma 2 text prefix makes a caption retrieve its own image best, on CPU.

The first N eval latents (256 px) are decoded by Primal and embedded as images at the 384 px budget (70 soft tokens,
`probe_vision.py`); each caption is embedded under each prefix of the model's prompt list and without one. Per prefix:
top-1 and top-5 retrieval of the image among the N by the caption, and the mean cosine to the own image against the
others. Usage: probe_prefix.py EVAL.parquet [N]
"""

import json
import sys

import pyarrow.parquet as pq
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from nanodit.data import _bf16
from nanodit.model import LATENT_CHANNELS
from train import load_decoder

EVAL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
PREFIXES = {
    "none": "",
    "query (search result)": "task: search result | query: ",
    "document": "title: none | text: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "sentence similarity": "task: sentence similarity | query: ",
}
torch.set_grad_enabled(False)

table = pq.read_table(EVAL, columns=["caption", "latent_256"]).slice(0, N)
captions = table["caption"].to_pylist()
latents = _bf16(table["latent_256"], LATENT_CHANNELS, 128, 128).float()
decoder = load_decoder(torch.device("cpu"))
images = torch.cat([decoder.decode(latents[i : i + 16]).clamp(-1, 1) for i in range(0, N, 16)])
pils = [Image.fromarray(((im + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()) for im in images]

st = SentenceTransformer("google/embeddinggemma-2", device="cpu", config_kwargs={"audio_config": None}, model_kwargs={"dtype": torch.float32})
st[0].processor.image_processor.max_soft_tokens = 70
image_emb = st.encode(pils, convert_to_tensor=True, normalize_embeddings=True, batch_size=16)
for name, prefix in PREFIXES.items():
    text = st.encode([prefix + c for c in captions], convert_to_tensor=True, normalize_embeddings=True, batch_size=16)
    sim = text @ image_emb.T  # caption i against every image
    own = sim.diag()
    rank = (sim > own[:, None]).sum(1)  # images scoring above the own one
    print(json.dumps({"prefix": name, "rows": N, "top1": (rank == 0).float().mean().item(), "top5": (rank < 5).float().mean().item(),
                      "own": own.mean().item(), "other": ((sim.sum(1) - own) / (N - 1)).mean().item()}), flush=True)
