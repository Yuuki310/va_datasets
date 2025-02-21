import os
import glob
import json
from pathlib import Path
from datasets_turntaking.utils import repo_root


lang = "eng"
dataset = "Callhome"
audio_dir = os.path.join(repo_root(), "data", dataset, lang, "data")

if __name__ == "__main__":
    path_list = glob.glob(os.path.join(audio_dir, "*.wav"))
    datalist = {}

    for audio_path in path_list:
        session = Path(audio_path).stem
        print(audio_path)
        relpath = os.path.relpath(audio_path, repo_root())
        datalist[session] = relpath
        json_path = os.path.join(os.path.dirname(__file__), "files/relative_audio_path.json")
        fw = open(json_path, 'w')
        json.dump(datalist, fw, indent=4)


