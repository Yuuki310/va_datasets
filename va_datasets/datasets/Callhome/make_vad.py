import os
import re
import glob
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

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

lang = "zho"
dataset = "Callhome"

if __name__ == "__main__":
    dataset_dir = os.path.join(repo_root(), "data", dataset+ "_" + lang)

    audio_dir = os.path.join(dataset_dir, "audio")
    text_dir = os.path.join(dataset_dir, "process", "transcript")
    vad_dir = os.path.join(dataset_dir, "vad")
    os.makedirs(vad_dir, exist_ok=True)

    data = []
    paths = load_data(audio_dir, text_dir)

    for path in tqdm(paths):
        session = Path(path["audio_path"]).stem

    
        audio, sr = sf.read(path["audio_path"])
        length = audio.shape[0] / sr

        _, _, dialog = load_utterances(path["text"])
        vad = extract_vad(dialog, length)

        encode_data = json.dumps(vad, indent=4)
        vad_path = os.path.join(vad_dir, f"{session}.json") 
        with open(vad_path, "w") as f:
            json.dump(vad, f, indent=4)

        data.append(
            {   
                "audio_path": str(path["audio_path"]),
                "vad_path": vad_path,
            }
        )
    new_df = pd.DataFrame(data)
    new_df.to_csv(os.path.join(dataset_dir, "audio_vad.csv"), index=False)
    