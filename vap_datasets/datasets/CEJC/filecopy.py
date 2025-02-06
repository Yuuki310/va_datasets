import os
import re
import glob
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_data,
    load_utterances,
    extract_vad,
)
from datasets_turntaking.utils import (
    repo_root,
    read_txt,
    read_json,
)
import shutil

dataset = "CEJC2021"


if __name__ == "__main__":
    csv_path = "/data/group1/z40351r/datasets_turntaking/data/CEJC2021/datapath/CEJCs/phone.csv"
    df = pd.read_csv(csv_path)
    data_dir = os.path.join(repo_root(), "data", dataset, "data")
    data = []
    for index, row in df.iterrows():
        new_dir = "/data/group1/z40351r/datasets_turntaking/data/CEJC2021/audio"
        first_audio_path = row["first_audio_path"]
        second_audio_path = row["second_audio_path"]
        shutil.copy(first_audio_path, new_dir)
        shutil.copy(second_audio_path, new_dir)
        

        data.append(
            {   
                "first_audio_path": str(os.path.join(new_dir, os.path.basename(first_audio_path))),
                "second_audio_path": str(os.path.join(new_dir, os.path.basename(second_audio_path))),
                "vad_path": row["vad_path"],
            }
        )

    new_df = pd.DataFrame(data)
    new_df.to_csv(os.path.join("/data/group1/z40351r/datasets_turntaking/data/CEJC2021/datapath/phone.csv"), index=False)
    


