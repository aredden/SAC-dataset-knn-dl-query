import tarfile
from typing import List
from PIL import Image
from glob import glob
from loguru import logger
from ray.util import ActorPool
import ray
import clip
import torch
import tqdm
from config import (
    tarfile_path,
    num_gpus,
    model_name,
    batch_size,
    buffer_size,
    result_embeds_filename,
)


@ray.remote(num_gpus=1)
class InferenceProcess:
    def __init__(self, model_name="ViT-B/32"):
        self.model, self.prep = clip.load(model_name, device="cuda")
        self.model = self.model.eval().requires_grad_(False)

    def generate(self, images: List[Image.Image], names: List[str], tarfiles: str):
        prepped_images = torch.cat(
            [self.prep(image).unsqueeze(0) for image in images], dim=0
        ).cuda()
        with torch.no_grad():
            embeds = self.model.encode_image(prepped_images).cpu().numpy()
        return [
            {"name": name, "tarfile": tarfile, "embed": embed}
            for name, tarfile, embed in zip(names, tarfiles, embeds)
        ]


class DataLoader:
    def __init__(self, tarpaths: List[str], batch_size=32, buffer_size=32) -> None:
        self.current_tar_index = 0
        self.current_list = []
        self.tarpaths = tarpaths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.empty = False

    def _reload(self, batch_size=32, buffer_size=32):
        if self.current_tar_index >= len(self.tarpaths):
            self.empty = True
            return False
        while len(
            self.current_list
        ) < buffer_size + batch_size and not self.current_tar_index >= len(
            self.tarpaths
        ):
            file = tarfile.open(self.tarpaths[self.current_tar_index])
            tmp_imgs = [
                (
                    Image.open(file.extractfile(m)),
                    m.name.split("/")[-1]
                    .replace(".png", "")
                    .replace(".", "")
                    .replace(",", "")
                    .replace("'", ""),
                    self.tarpaths[self.current_tar_index],
                )
                for m in file.getmembers()
            ]
            self.current_tar_index += 1
            self.current_list.extend(tmp_imgs)

    def iterate(self, batch_size=None, buffer_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if buffer_size is None:
            buffer_size = self.buffer_size
        while not self.empty:
            if len(self.current_list) < batch_size + buffer_size:
                self._reload()
            batch = self.current_list[:batch_size]
            self.current_list = self.current_list[batch_size:]
            yield batch


workers = [InferenceProcess.remote(model_name=model_name) for _ in range(num_gpus)]


pool = ActorPool(workers)
output_tarfiles = glob(f"{tarfile_path}/*.tar")
image_dloader = DataLoader(
    output_tarfiles, batch_size=batch_size, buffer_size=buffer_size
)
results = pool.map(
    lambda a, vals: a.generate.remote(
        images=[v[0] for v in vals],
        names=[v[1] for v in vals],
        tarfiles=[v[2] for v in vals],
    ),
    image_dloader.iterate(),
)
all_results = []

for result in tqdm.tqdm(results):
    logger.info(f"Appending {len(result)} results")
    all_results.extend(result)

torch.save(all_results, result_embeds_filename)
