from faiss import read_index, Index2Layer
import clip
import numpy as np
from time import time
from config import (
    result_embeds_filename,
    model_name,
)
import torch

text_query = "I am a dog"
data = torch.load(result_embeds_filename)
labels, tarfiles = [d["name"] for d in data], [d["tarfile"] for d in data]
model, prep = clip.load(model_name, device="cuda")
model = model.eval().requires_grad_(False)
embed = clip.tokenize(text_query).cuda()
q = model.encode_text(embed).cpu().numpy().astype(np.float32)

index: Index2Layer = read_index("./knn.index")
s = time()
indices = index.search(q, 100, labels=labels)
t = time() - s
print(indices, "Total time seconds:", t)
