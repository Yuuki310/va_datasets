# coding: UTF-8
import datetime
import json
import os
import random
import time
import warnings
from os.path import join, dirname, basename, exists
from pprint import pprint
import matplotlib.pyplot as plt

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf 
from tqdm import tqdm


from vap_datasets.utils.utils import (
    repo_root,
    load_waveform,
)
from vap_datasets.plot.utils import(
    plot_stereo,
    load_vad,
    plot_compare
)

warnings.simplefilter("ignore")
from vap_datasets.datamodule.datamodule import VAPDataModule



def main() -> None:
    dataset = "CEJC"
    session = "K005_031"
    audio_path = os.path.join(repo_root(), "data", dataset, "audio", session + ".wav")

    img_dir = os.path.join(repo_root(), "vap_datasets", "plot")
    os.makedirs(img_dir, exist_ok=True)


    vad_path = os.path.join(repo_root(), "data", dataset, "vad", session + ".json")
    webrtc_path = os.path.join(repo_root(), "data", dataset, "webrtc", session + ".json")
    output_path = os.path.join(img_dir, f"{os.path.basename(audio_path).replace('wav', 'png')}")


    # print(dm)
    # print("VAPDataModule: ", len(dm.test_dset))
    waveform = load_waveform(audio_path)[0]
    va = load_vad(vad_path)
    webrtc = load_vad(webrtc_path)
    fig, ax = plot_compare(
        waveform, [va, webrtc], figsize=(80, 20), start=40, end=60
    )


    fig.savefig(output_path)

            
        


if __name__ == "__main__":
    main()
