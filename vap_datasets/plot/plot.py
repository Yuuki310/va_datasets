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
    load_waveform,
)
from vap_datasets.plot.utils import(
    plot_stereo,
    load_vad    
)

warnings.simplefilter("ignore")
from vap_datasets.datamodule.datamodule import VAPDataModule



def main() -> None:
    dataset = "CallhomeENG"
    session = "4941"
    label_subset="webrtc"
    audio_path = os.path.join(repo_root(), "data", dataset, "audio", session + ".wav")
    img_dir = os.path.join(repo_root(), "vap_datasets", "plot")
    os.makedirs(img_dir, exist_ok=True)
    vad_path = os.path.join(repo_root(), "data", dataset, "labels", label_subset, session + ".json")
    output_path = os.path.join(img_dir, f"{os.path.basename(audio_path).replace('.wav', f'_{label_subset}.png')}")


    # print(dm)
    # print("VAPDataModule: ", len(dm.test_dset))
    waveform = load_waveform(audio_path)[0]
    va = load_vad(vad_path)
    fig, ax = plot_stereo(
        waveform, va, figsize=(40, 20)
    )
    ax[0].set_xlim([188, 198])
    ax[1].set_xlim([188, 198])


    fig.savefig(output_path)

            
        


if __name__ == "__main__":
    main()
