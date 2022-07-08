from braceexpand import braceexpand
import os, wget, tqdm, ray
from typing import List
from loguru import logger
from config import tarfile_path

urls = "https://s3.us-west-1.wasabisys.com/simulacrabot/simulacra-aesthetic-captions/sac-{000000..000086}.tar"
urls = list(braceexpand(urls))

assert len(urls) == 87, "Expected 87 urls, got {}".format(len(urls))
if not os.path.exists(tarfile_path):
    os.makedirs(tarfile_path, exist_ok=True)


@ray.remote(num_cpus=0.5)
def download_tar(url):
    output_dir = os.path.join(tarfile_path, url.split("/")[-1])
    if not os.path.exists(output_dir):
        logger.info(f"Downloading {url} to {output_dir}")
        wget.download(url, output_dir)
        logger.info("Downloaded {}".format(url))
    else:
        logger.info(f"{url} already downloaded")
    return True


def into_iterator(tasks: List[ray.TaskID]):
    while tasks:
        done, tasks = ray.wait(tasks)
        yield ray.get(done[0])


tasks_list = [download_tar.remote(url=url) for url in urls]

idx = 0
for i in tqdm.tqdm(into_iterator(tasks_list), total=len(tasks_list)):
    logger.info(f"Completed task {idx}")
    idx += 1
