import os 
import soundfile as sf
import json
import pandas as pd
import shutil
from utils import load_data, extract_vad
from vap_datasets.utils.utils import (
    repo_root,
)
from tqdm import tqdm


if __name__ == "__main__":
    dataset = "CSJ"
    source_dir = os.path.join(repo_root(), "source", dataset)
    source_audio_dir = os.path.join(source_dir, "WAV")
    source_transcript_dir = os.path.join(source_dir, "TRN", "Form2")

    dataset_dir = os.path.join(repo_root(), "data", dataset)
    audio_dir = os.path.join(dataset_dir, "audio")
    vad_dir = os.path.join(dataset_dir, "vad")

    path_list = load_data(source_audio_dir, source_transcript_dir)

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(vad_dir, exist_ok=True)
    data = []

    for path in tqdm(path_list):
        session = os.path.splitext(os.path.basename(path["audio"]))[0]

        shutil.copy(path["audio"],audio_dir)
        vad = extract_vad(path["transcript"])
        encode_data = json.dumps(vad, indent=4)
        vad_path = os.path.join(vad_dir, f"{session}.json")
        
        with open(vad_path, "w") as f:
            json.dump(vad, f, indent=4)

        data.append(
            {   
                "session": session
            }
        )
    os.makedirs(os.path.join(dataset_dir, "split"), exist_ok=True)
    new_df = pd.DataFrame(data)
    new_df.to_csv(os.path.join(dataset_dir, "split/overview.csv"), index=False)
    
        