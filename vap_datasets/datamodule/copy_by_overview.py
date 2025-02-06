from os.path import join, isfile
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
from argparse import ArgumentParser
from vap_datasets.utils.utils import repo_root

if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--new_dataset", type=str)
    args = parser.parse_args()

    new_dataset_dir = os.path.join(repo_root(), "data", args.new_dataset)
    csv_path = os.path.join(new_dataset_dir, "split/overview.csv")
    df = pd.read_csv(csv_path)

    new_audio_dir = os.path.join(new_dataset_dir, "audio")
    new_vad_dir = os.path.join(new_dataset_dir, "vad")
    
    os.makedirs(new_audio_dir, exist_ok=True)
    os.makedirs(new_vad_dir, exist_ok=True)
    
    for index, row in df.iterrows():
        session = row["session"]
        audio_path = os.path.join(repo_root(), "data", args.dataset, "audio", session + ".wav")
        vad_path = os.path.join(repo_root(), "data", args.dataset, "vad", session + ".json")
        shutil.copy(audio_path, new_audio_dir)
        shutil.copy(vad_path, new_vad_dir)
