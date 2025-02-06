import torch
from torch.utils.data import Dataset
from pathlib import Path
from os.path import dirname
import os
import pandas as pd

import tqdm

from vap_datasets.datamodule.datamodule import force_correct_nsamples
from vap.utils.audio import load_waveform
from vap.utils.utils import vad_list_to_onehot, read_json
from vap.events.events import HoldShift

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True
    )
    parser.add_argument(
        "--out_dir",
        required=True
    )
    args = parser.parse_args()

    dfs = []
    train_file = os.path.join("data", args.dataset, "path", args.out_dir, "train_hs.csv")
    test_file = os.path.join("data", args.dataset, "path", args.out_dir, "test_hs.csv")
    val_file = os.path.join("data", args.dataset, "path", args.out_dir, "val_hs.csv")
    
    shift_file = os.path.join("data", args.dataset, "path", args.out_dir, "shifts.csv")
    hold_file = os.path.join("data", args.dataset, "path", args.out_dir, "holds.csv")

    for file in [train_file, test_file, val_file]:
        df = pd.read_csv(file)
        dfs.append(df)

  
    c = pd.concat(dfs, ignore_index=True)
    c_shift = c[c["label"].isin(["shift", "overlap_shift", "notime_shift"])]
    c_shift.to_csv(shift_file, index=False)
    
    c_hold = c[c["label"].isin(["hold"])]
    c_hold.to_csv(hold_file, index=False)
    
    print(len(c_shift[(c_shift["tfo"] <= 0.1) & (c_shift["tfo"] > 0.0)]))
    print(len(c_shift[(c_shift["tfo"] <= 0.2) & (c_shift["tfo"] > 0.1)]))
    print(len(c_shift[(c_shift["tfo"] <= 0.0) & (c_shift["tfo"] > -0.1)]))

    # fig = plt.figure(figsize = (12,9), facecolor='white')
    # bins = np.arange(-3.05, 3.05, 0.1)

    # plt.xlabel("Turn-shift gap [sec]")
    # plt.ylabel("Count")
    # plt.grid([-2,-1,0,1,2])
    # plt.hist(c_shift["tfo"], bins=bins, ec='black')
    # fig.savefig(shift_file.replace(".csv", ".png"))
    
    fig, ax= plt.subplots(figsize = (16,10))
    fig.set_facecolor("white")
    ax.set_facecolor("lightblue")
    ax.patch.set_alpha(0.6)
    bins = np.arange(-3.05, 3.05, 0.1)
    plt.xlabel("Turn-shift gap [sec]", fontsize=28, labelpad=10)
    plt.ylabel("Count", fontsize=28, labelpad=10)
    plt.tick_params(labelsize=20)
    plt.grid([-2,-1,0,1,2], color="white", linewidth=2)
    plt.hist(c_shift["tfo"], bins=bins, ec='black')
    ax.set_axisbelow(True)
    ax.get_xaxis().set_tick_params(pad=10)
    ax.get_yaxis().set_tick_params(pad=10)

    fig.savefig(shift_file.replace(".csv", ".png"))
    
    # c_hold = c[c["label"].isin(["hold"])]
    # c_hold.to_csv(hold_file, index=False)
    
    # fig = plt.figure(figsize = (12,9), facecolor='white')
    # bins = np.arange(0, 5, 0.1)

    # plt.hist(c_hold["tfo"], bins=bins, ec='black')
    # plt.xlabel("Turn-shift gap [sec]")
    # plt.ylabel("Count")
    # fig.savefig(hold_file.replace(".csv", ".png"))
