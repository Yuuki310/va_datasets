import os
import re
import glob
from pathlib import Path
from tqdm import tqdm
import json

from datasets_turntaking.utils import (
    repo_root,
    read_txt,
    read_json,
)
from pydub import AudioSegment
from pathlib import Path
import shutil
import librosa
import soundfile as sf
if __name__ == "__main__":

    data_dir = os.path.join(repo_root(), "data", "BTSJ", "data")
    origin_dir = os.path.join(repo_root(), "data", "BTSJ", "3.コーパスのトランスクリプト・音声")

    sessions = glob.glob(os.path.join(origin_dir, "**/*.mp3"), recursive=True)

    print(sessions)
    for session in sessions:
        session_name = Path(session).stem
        print(session_name)
        in_audio = AudioSegment.from_mp3(session)
        out_path = os.path.join(data_dir, session_name + ".wav")
        in_audio.export(out_path, format="wav")
        y, sr = librosa.core.load(out_path, sr=16000, mono=False)
        sf.write(out_path, y.T, sr)

        transcript_path = glob.glob(os.path.join(origin_dir, "**", session_name + ".xlsx"), recursive=True)[0]
        transcript_new_path = os.path.join(data_dir, session_name + ".xlsx")
        shutil.copy(transcript_path, transcript_new_path)
