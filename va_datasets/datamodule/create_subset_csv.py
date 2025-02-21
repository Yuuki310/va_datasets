from os.path import join, isfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os
from argparse import ArgumentParser

import sys
from vap_datasets.utils.utils import repo_root

if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--audio_dir", type=str, default="data/audio")
    parser.add_argument("--label_dir", type=str, default="data/labels/original")
    parser.add_argument("--out_dir", type=str, default="data/default")    
    args = parser.parse_args()
    
    dataset = args.dataset

    audio_dir = args.audio_dir
    label_dir = args.label_dir
    out_dir = args.out_dir

    for split in ["train", "test", "val"]:
        df_path = join(repo_root(), f"data/{dataset}/split/{split}.csv")
        df = pd.read_csv(df_path, dtype=str)
        data = []
        for index, row in tqdm(df.iterrows()):
            name = str(row["session"])
            audio_path = join(repo_root(), audio_dir, f"{name}.wav")
            vad_path = join(repo_root(), label_dir, f"{name}.json")
            
            if not isfile(vad_path):
                print(f"Missing {vad_path}")
                continue
            data.append(
                {
                    "audio_path": str(audio_path),
                    "vad_path": str(vad_path),
                }
            )

        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f"{split}.csv")
        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False)
        print("Saved -> ", out_path)

