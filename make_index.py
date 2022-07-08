import autofaiss as af
import numpy as np
import torch
from config import (
    result_embeds_filename,
    index_nb_cores,
    index_current_mem_available,
    index_max_memory_usage,
)

data = torch.load(result_embeds_filename)
embeds = np.array([d["embed"] for d in data])

af.build_index(
    embeds,
    nb_cores=index_nb_cores,
    current_memory_available=index_current_mem_available,
    max_index_memory_usage=index_max_memory_usage,
)
