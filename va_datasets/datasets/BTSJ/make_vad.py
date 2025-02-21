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

dataset = "BTSJ"

if __name__ == "__main__":
    data_dir = os.path.join(repo_root(), "data", dataset, "data")
    vad_dir = os.path.join(repo_root(), "data", dataset, "vad")
    os.makedirs(vad_dir, exist_ok=True)

    sessions = glob.glob(os.path.join(data_dir, "*.wav"))
    data = []
    
    for audio_path in sessions:
        session = os.path.splitext(os.path.basename(audio_path))[0]
        first_speaker_id = session.split("-")[2]
        second_speaker_id = session.split("-")[3]

        csv_path = os.path.join(data_dir, f"{session}.xlsx")
        dialog = load_utterances(csv_path, first_speaker_id, second_speaker_id)
        vad = extract_vad(dialog)
        vad_path = os.path.join(vad_dir, f"{session}.json") 

        with open(vad_path, "w") as f:
            json.dump(vad, f, indent=4)

        data.append(
            {   
                "audio_path": [str(audio_path)],
                "vad_path": vad_path,
            }
        )

    new_df = pd.DataFrame(data)
    new_df.to_csv(os.path.join(repo_root(), "datasets_turntaking", "datasets", dataset, "files", "audio_vad.csv"), index=False)
    