import os
import torch

num_gpus = torch.cuda.device_count()
model_name = "ViT-L/14"
tarfile_path = "outputs"
batch_size = 128
buffer_size = 512
result_embeds_filename = "results.pth"
index_nb_cores = os.cpu_count()
index_current_mem_available = "32GB"
index_max_memory_usage = "96GB"
