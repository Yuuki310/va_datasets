import os
import re
import glob
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from vap_datasets.utils.utils import (
    repo_root,
    read_txt,
    read_json,
)
import soundfile as sf
import numpy as np
import shutil

def extract_vad(textgrid):
    vad = []
    with open(textgrid, encoding="utf-16") as f:
        textgrid = f.readlines()
        text = []
        for i, row in enumerate(textgrid):
            if "intervals" in row:
                xmin = float(re.search(r'\d+(\.\d+)?', textgrid[i+1].strip()).group())
                xmax = float(re.search(r'\d+(\.\d+)?', textgrid[i+2].strip()).group())
                match = re.search(r'"(.*?)"', textgrid[i+3].strip())
                if match:
                    content = match.group(1)
                else:
                    content = ""
                text.append([xmin, xmax, content])
        for utter in text:
            if utter[2] == "":
                continue
            else:
                vad.append([utter[0], utter[1]])
        return vad
    

if __name__ == "__main__":
    dataset = "MDT"
    source_dir = os.path.join(repo_root(), "source", dataset)

    audio_dir = os.path.join(repo_root(), "data", dataset, "audio")
    vad_dir = os.path.join(repo_root(), "data", dataset, "vad")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(vad_dir, exist_ok=True)
    tescom_list = glob.glob(os.path.join(source_dir, "WAV", "*/MDT_Conversation_*_TESCOM.wav"))
    textgrid_list = glob.glob(os.path.join(source_dir, "WAV", "*/MDT_Conversation_*_SPK*.TextGrid"))
    
    data = []
    for tescom_path in tescom_list:
        session = os.path.basename(os.path.dirname(tescom_path))
        speakers_path = glob.glob(os.path.join(source_dir, "WAV", session, f"{session}_SPK*.wav"))
        print(session)
        print(tescom_path)
        print(speakers_path)
        sorted(speakers_path)
        speakers_path.sort()
        l_audio, sr = sf.read(speakers_path[0])
        r_audio, _ = sf.read(speakers_path[1])
        
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

        vad_path = os.path.join(vad_dir, session + ".json")

        textgrid_list = sorted(glob.glob(os.path.join(source_dir, "TXT", session + "*.TextGrid")))
        vad0 = extract_vad(textgrid_list[0])
        vad1 = extract_vad(textgrid_list[1])
        vad = [vad0, vad1]

        encode_data = json.dumps(vad, indent=4)
        with open(vad_path, "w") as f:
            json.dump(vad, f, indent=4)

        data.append({"session": session})

    new_df = pd.DataFrame(data)
    os.makedirs(os.path.join(repo_root(), "data", dataset, "split"), exist_ok=True)
    new_df.to_csv(os.path.join(repo_root(), "data", dataset, "split", "overview.csv"), index=False)
    new_df.to_csv(os.path.join(repo_root(), "data", dataset, "split", "test.csv"), index=False)