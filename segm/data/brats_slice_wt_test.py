#versione implementata ispirandomi a https://www.codestudyblog.com/cs2112pyc/1230061614.html

from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from segm.data.utils import IGNORE_LABEL #new
brats_slice_CONFIG_PATH = Path(__file__).parent / "config" / "brats_slice_wt_test.py"
brats_slice_CATS_PATH = Path(__file__).parent / "config" / "brats_slice_wt.yml"


class BratsSliceWTDatasetTest(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
    #def __init__(self, image_size=180, crop_size=180, split='train', **kwargs):
        print("CLASS DEFINITION ARGS: ")
        print(kwargs)
        print("image_size = {}".format(image_size))
        print("crop_size = {}".format(crop_size))
        print("split = {}".format(split))
        #print("normalization = {}".format(normalization))
        super().__init__(
            image_size,
            crop_size,
            split,
            brats_slice_CONFIG_PATH,
            #normalization = normalization,
            **kwargs,
        )
        self.names, self.colors = utils.dataset_cat_description(brats_slice_CATS_PATH)
        self.n_cls = 2#3
        self.ignore_label = None#IGNORE_LABEL
        self.reduce_zero_label = False #True

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir) / "brats_slice"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path / ""#"ADEChallengeData2016"
        elif self.split == "trainval":
            config.data.trainval.data_root = path / ""#"ADEChallengeData2016"
        elif self.split == "val":
            config.data.val.data_root = path / ""#"ADEChallengeData2016"
        elif self.split == "test":
            config.data.test.data_root = path / "release_test"
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels + 1
