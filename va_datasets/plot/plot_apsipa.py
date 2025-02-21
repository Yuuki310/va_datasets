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
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    label = "webrtc"
    data_1 = {"dataset":"Switchboard", "session": "sw04319", "label":label, "start":88, "end":98}
    data_2 = {"dataset":"CallhomeENG", "session": "4941", "label":label, "start":188, "end":198}

    img_dir = os.path.join(repo_root(), "vap_datasets", "plot")
    os.makedirs(img_dir, exist_ok=True)

    output_path = os.path.join(img_dir, f"apsipa_{label}.png")

    result_fig, result_ax = plt.subplots(2, 1, figsize=(40, 20))



    for i, data in enumerate([data_1, data_2]):
        dataset = data["dataset"]
        session = data["session"]
        label=data["label"]
        audio_path = os.path.join(repo_root(), "data", dataset, "audio", session + ".wav")
        vad_path = os.path.join(repo_root(), "data", dataset, "labels", label, session + ".json")
        output_data_path = os.path.join(img_dir, f"{os.path.basename(audio_path).replace('.wav', f'_{label}.png')}")
        waveform = load_waveform(audio_path)[0]
        va = load_vad(vad_path)
        fig, ax = plot_stereo(
            waveform, va, figsize=(40, 20)
        )
        ax[0].set_xlim([data["start"], data["end"]])
        fig.savefig(output_data_path)

        for line in ax[0].get_lines():
            result_ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())
        result_ax[i].set_xlim([data["start"], data["end"]])

    plt.subplots_adjust(
        left=0.08, bottom=None, right=None, top=None, wspace=None, hspace=0.04
    )
    result_fig.savefig(output_path)

            
        


if __name__ == "__main__":
    main()
