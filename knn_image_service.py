import torch, numpy as np, tarfile, clip, io, config, cv2, matplotlib.pyplot as plt
from typing import List
from faiss import read_index
from PIL import Image
import io
import config


class KnnImageService:
    """
    Caching knn query service using a faiss KNN index.
    """

    def __init__(self, verbose=True):
        self.data = torch.load(config.result_embeds_filename)
        self.model, self.prep = clip.load(config.model_name, device="cuda")
        self.model = self.model.eval().requires_grad_(False)
        self.index = read_index("./knn.index")
        self.tars_cache = {}
        self.verbose = verbose

    def query(
        self,
        text_query: str,
        nb_results: int = 100,
        nrows: int = 4,
        return_pil: bool = True,
    ):
        """
        Queries the index for the given text query and returns the top nb_results.
        """
        embed = clip.tokenize(text_query).cuda()
        with torch.no_grad():
            q = self.model.encode_text(embed).cpu().numpy().astype(np.float32)
        indices = self.index.search(q, nb_results)[-1]
        if return_pil:
            return Image.fromarray(self.assemble_grid(indices, nrows))
        else:
            return self.assemble_grid(indices, nrows)

    def assemble_grid(self, query_indicies: List[int], nrows: int = 4):
        """
        Assembles a grid of images from the top nb_results of the query_indicies.
        """
        images = []
        if self.verbose:
            print("Assembling grid...", query_indicies)
        for i in query_indicies[0]:
            item = self.data[i]
            images.append(self.extract_image(item))
        return self.image_grid(images, nrows=nrows)

    def extract_image(self, item):
        """
        Reads the image from it's associated tarfile and returns it, caching the tarfile for future use.
        """
        tar = item["tarfile"]
        if tar not in self.tars_cache:
            data_tar = tarfile.open(tar)
            members = data_tar
            all_names = [
                m.name.split("/")[-1]
                .replace(".png", "")
                .replace(".", "")
                .replace(",", "")
                .replace("'", "")
                for m in members
            ]
            self.tars_cache[tar] = {
                "tar": data_tar,
                "names": all_names,
                "ids": [m.name.split("/")[-1].split("_")[0] for m in members],
            }
        else:
            members = self.tars_cache[tar]["tar"]
            all_names = self.tars_cache[tar]["names"]
        index = all_names.index(item["name"])
        member = members.getmembers()[index]
        if self.verbose:
            print(member)
        im = np.array(
            Image.open(io.BytesIO(members.extractfile(member).read())).convert("RGB")
        )
        return im

    def image_grid(self, images: List[np.ndarray], nrows: int):
        """
        Creates a grid of images with 'nrows' number of image rows, images must all be the same shape.
        """
        assert len(images) > nrows, "Must have more images than rows"
        ncols = 4
        nrows = int(np.ceil(len(images) / ncols))
        normal_size = 512
        for i, x in enumerate(images):
            if x.shape[0] != normal_size or x.shape[1] != normal_size:
                images[i] = cv2.resize(
                    x, (normal_size, normal_size), interpolation=cv2.INTER_LANCZOS4
                )
        images_size = images[0].shape[:2]
        grid_img = np.zeros(
            (nrows * images_size[0], ncols * images_size[1], 3), dtype=np.uint8
        )
        for i, img in enumerate(images):
            r = int(i / ncols)
            c = i % ncols
            grid_img[
                r * images_size[0] : (r + 1) * images_size[0],
                c * images_size[1] : (c + 1) * images_size[1],
            ] = img
        return grid_img
