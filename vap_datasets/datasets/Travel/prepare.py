import glob
import os
import glob
from tqdm import tqdm
import numpy as np
import soundfile as sf
import pandas as pd
import json
from vap_datasets.utils.utils import (
    repo_root,
)
from utils import (
    extract_vad
)



    
if __name__ == "__main__":

    dataset = "Travel"

    audio_dir = os.path.join(repo_root(), "data", dataset, "audio")
    vad_dir = os.path.join(repo_root(), "data", dataset, "vad")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(vad_dir, exist_ok=True)

    operator_paths = glob.glob(os.path.join(repo_root(), "source", dataset, "wav/*_operator.wav"))
    count = 0
    data = []
    for operator_path in tqdm(operator_paths):
        session = os.path.basename(operator_path)[:7]
        if session == "315_1_2":
            continue
        user_path = os.path.join(os.path.dirname(operator_path), f"{session}_user.wav")

        l_audio, sr = sf.read(operator_path)
        r_audio, _ = sf.read(user_path)
        
        audio_path = os.path.join(audio_dir, session + ".wav")       
        if len(l_audio) != len(r_audio):
            continue
            # if len(l_audio) > len(r_audio):
            #     l_audio = l_audio[:len(r_audio)]
            # else:
            #     r_audio = r_audio[:len(l_audio)]
            # print("length error")
            # print(operator_path, user_path)

        stereo_audio = np.column_stack((l_audio, r_audio))
        sf.write(audio_path, stereo_audio, sr)
    
        annotation_path = os.path.join(repo_root(), "source", dataset, "annotations", session + "_full.json")
        vad = extract_vad(annotation_path)        
        vad_path = os.path.join(vad_dir, session + ".json")
        encode_data = json.dumps(vad, indent=4)
        with open(vad_path, "w") as f:
            json.dump(vad, f, indent=4)

        data.append({"session": session})

    new_df = pd.DataFrame(data)
    os.makedirs(os.path.join(repo_root(), "data", dataset, "split"), exist_ok=True)
    new_df.to_csv(os.path.join(repo_root(), "data", dataset, "split", "overview.csv"), index=False)
    



