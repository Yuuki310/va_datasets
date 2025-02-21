# coding: UTF-8
import datetime
import json
import os
import random
import time
import warnings
from os.path import join, dirname, basename, exists
from pprint import pprint

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf 
from tqdm import tqdm


from vap_datasets.utils.utils import (
    repo_root,
    load_waveform
)
from vap_datasets.plot.utils import(
    plot_stereo,
    onehot_to_segment
)

warnings.simplefilter("ignore")
from vap_datasets.datamodule.datamodule import VAPDataModule



def main(args) -> None:
    dataset = args.dataset_name
    subset = args.subset
    img_dir = os.path.join(repo_root(), "data", dataset, "img", subset)
    os.makedirs(img_dir, exist_ok=True)
    dm = VAPDataModule(
        datasets=[dataset],
        subsets=[subset],
        mono = False,
        mode = "stereo",
        sample_rate=8000,
    )
    dm.prepare_data()
    dm.setup()
    dm.setup("test")
    # print(dm)
    # print("VAPDataModule: ", len(dm.test_dset))
    dloader = dm.test_dataloader()
    for ii, batch in tqdm(
        enumerate(dloader),
        total=len(dloader)
    ):
        waveform = batch["waveform"][0].cpu()
        va = batch["vad"].to("cuda")[0].cpu()
        print(va)
        va_label_0 = onehot_to_segment(va[:,0])
        va_label_1 = onehot_to_segment(va[:,1])


        fig, ax = plot_stereo(
            waveform, [va_label_0, va_label_1], figsize=(80, 20)
        )

        output_path = os.path.join(img_dir, f"{ii:04}.png")
        fig.savefig(output_path)

            
        


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--subset", type=str)
    args = parser.parse_args()

    main(args)
