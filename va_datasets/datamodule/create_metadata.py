from os.path import join, isfile
import os
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from vap_datasets.utils.utils import repo_root
import soundfile as sf

from vap_datasets.datamodule.create_sliding_window_dset import get_vad_list_lims
from vap_datasets.utils.utils import read_json

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    split_dir = os.path.join(repo_root(), "data", args.dataset, "split")
    overview_path = os.path.join(split_dir, "overview.csv")
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")

    audio_dir = os.path.join(repo_root(), "data", args.dataset, "audio")
    vad_dir = os.path.join(repo_root(), "data", args.dataset, "webrtc")
        
    paths = {"train":train_path, "val":val_path, "test":test_path}
    metadata = {}


    for subset, path in tqdm(paths.items()):
        df = pd.read_csv(path, dtype=str)
        subset_durations = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            session = str(row["session"])
            audio_path = os.path.join(audio_dir, f"{session}.wav")
            vad_path = os.path.join(vad_dir, f"{session}.json")
            sound, sr = sf.read(audio_path)
            duration = len(sound) / sr
            subset_durations.append(duration)

            vad = read_json(vad_path)
            start, end = get_vad_list_lims(vad)
        metadata[subset] = {
            "duration" : sum(subset_durations)/3600,
            "average" : sum(subset_durations) / len(subset_durations),
            "data_num" : len(subset_durations)
        }        


    total_duration = sum(metadata[key]["duration"] for key in ["train", "test", "val"])
    total_num = sum(metadata[key]["data_num"] for key in ["train", "test", "val"])
    metadata["total"] = {
        "total" : total_duration,
        "data_num" : total_num
    }

    print(metadata)    
    out_path = join(repo_root(), "data", args.dataset, f"metadata.json")
    # df = pd.DataFrame(metadata)
    # df.to_csv(out_path)

    encode_data = json.dumps(metadata, indent=4)

    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=4)

        
