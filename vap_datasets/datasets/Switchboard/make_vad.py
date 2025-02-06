import os
import re
import json

from utils import read_txt

from datasets_turntaking.datasets.Switchboard.utils import (
    extract_dialog,
    extract_vad_list_from_words,
    remove_words_from_dialog,
)
from datasets_turntaking.utils import (
    read_txt,
    read_json,
    repo_root,
)

REL_AUDIO_PATH = os.path.join(
    repo_root(), "datasets_turntaking/datasets/Switchboard/split/relative_audio_path.json"
)
SPLIT_PATH = os.path.join(repo_root(), "datasets_turntaking/datasets/Switchboard/split")

vad_dir = os.path.join(repo_root(), "datasets_turntaking/datasets/Switchboard/vad")

if __name__ == "__main__":
    alignmets_path = "/data/group1/z40351r/datasets_turntaking/data/Switchboard/word_alignments"
    sess_2_rel_path = read_json(REL_AUDIO_PATH)


    train_sessions = read_txt(os.path.join(SPLIT_PATH, "train.txt"))
    val_sessions = read_txt(os.path.join(SPLIT_PATH, "val.txt"))
    test_sessions = read_txt(os.path.join(SPLIT_PATH, "test.txt"))
    
    os.makedirs(vad_dir, exist_ok=True)
    for sessions in [train_sessions, val_sessions, test_sessions]:
        for session in sessions:
            session = str(session)
            session_dir = os.path.join(
                alignmets_path, "swb_ms98_transcriptions", session[:2], session
            )
            dialog = extract_dialog(session, session_dir)
            vad = extract_vad_list_from_words(dialog)
            encode_data = json.dumps(vad, indent=4)
            
            with open(os.path.join(vad_dir, f"sw0{session}.json"), "w") as f:
                json.dump(vad, f, indent=4)
