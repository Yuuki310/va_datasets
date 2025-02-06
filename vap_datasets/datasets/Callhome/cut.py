import os 
import soundfile as sf
import json

from utils import load_data, get_speaker_num, load_utterances
from datasets_turntaking.utils import (
    read_txt,
    read_json,
    repo_root,
)

def cut_audio(path, start, end):
    newpath = os.path.join(out_dir,  os.path.basename(path))
    data, sr = sf.read(path)
    sf.write(newpath, data[int(start*sr/1000) : int(end*sr/1000)], sr)
    print(newpath)


if __name__ == "__main__":
    dataset = "Callhome"
    lang = "zho"
    sr = "8k"

    dataset_dir = os.path.join(repo_root(), "data", dataset+ "_" + lang)
    source_dir = os.path.join(dataset_dir, "full", sr)
    out_dir = os.path.join(dataset_dir, "audio", sr)
    text_dir = os.path.join(dataset_dir, "full" , "transcript")
    paths = load_data(source_dir, text_dir)

    os.makedirs(out_dir, exist_ok=True)
    datalist = {}
    for path in paths:
        if get_speaker_num(path["text"]) > 2: 
            print(f"Number of speakers is greater than 3 :  {path}")
            continue

        try:        
            start, end, utterances = load_utterances(path["text"])
            audio_path = path["audio_path"]
            cut_audio(audio_path, start, end)
        except:
            continue
        
        