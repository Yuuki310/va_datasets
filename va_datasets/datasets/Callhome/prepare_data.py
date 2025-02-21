import os
import re
import glob
from pathlib import Path
from tqdm import tqdm

from datasets_turntaking.utils import read_txt
from datasets_turntaking.utils import read_json

from utils import (
    load_utterances,
    extract_vad,
)
import json

lang = "zho"
vad_dir = "/data/group1/z40351r/datasets_turntaking/data/Callhome/" + lang + "/vad"
data_dir = "/data/group1/z40351r/datasets_turntaking/data/Callhome/" + lang

if __name__ == "__main__":
    sessions = glob.glob(os.path.join(data_dir, "data/*.wav"))
    os.makedirs(vad_dir, exist_ok=True)

    for session in tqdm(sessions):
        session = Path(session).stem
        trans_path = os.path.join(data_dir, "transcript", f"{session}.cha")
        _, _, dialog = load_utterances(trans_path)
        vad = extract_vad(dialog)
        encode_data = json.dumps(vad, indent=4)
        with open(os.path.join(vad_dir, f"{session}.json"), "w") as f:
            json.dump(vad, f, indent=4)
