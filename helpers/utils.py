import logging
import os
import pathlib

from beir.beir import util
from beir.beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


def get_dataset(dataset, out_dir="datasets") -> GenericDataLoader:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), out_dir)
    data_path = util.download_and_unzip(url, out_dir)

    return GenericDataLoader(data_path)
