import os 
import soundfile as sf
import json
import pandas as pd
import shutil
from utils import load_data, get_speaker_num, load_utterances, extract_vad, cut_audio
from vap_datasets.utils.utils import (
    read_txt,
    read_json,
    repo_root,
)


if __name__ == "__main__":
    dataset = "Callfriend"
    lang = "ENG"

    source_dir = os.path.join(repo_root(), "source", dataset + lang)
    source_audio_dir = os.path.join(source_dir, "wav")
    source_transcript_dir = os.path.join(source_dir, "transcript")
    dataset_dir = os.path.join(repo_root(), "data", dataset + lang)
    audio_dir = os.path.join(dataset_dir, "audio")
    vad_dir = os.path.join(dataset_dir, "vad")
    paths = load_data(source_audio_dir, source_transcript_dir)

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(vad_dir, exist_ok=True)

    data = []
    for path in paths:
        if get_speaker_num(path["text"]) > 2: 
            print(f"Number of speakers is greater than 3 :  {path}")
            continue
        print(path)
        session = os.path.splitext(os.path.basename(path["audio_path"]))[0]
        if lang == "ENG" and session == "6931":
            continue

        audio, sr = sf.read(path["audio_path"])
        length = audio.shape[0] / sr
        # shutil.copy(path["audio_path"], audio_dir)

        utterances = load_utterances(path["text"])
        vad = extract_vad(utterances, length)
        encode_data = json.dumps(vad, indent=4)
        vad_path = os.path.join(vad_dir, f"{session}.json")
        
        with open(vad_path, "w") as f:
            json.dump(vad, f, indent=4)

        s_end = max(vad[0][-1][-1], vad[1][-1][-1])
        out_path = os.path.join(audio_dir, session + ".wav")
        cut_audio(path["audio_path"], out_path, 0, s_end)
        
        data.append(
            {   
                "session": session
            }
        )
    os.makedirs(os.path.join(dataset_dir, "split"), exist_ok=True)
    new_df = pd.DataFrame(data)
    new_df.to_csv(os.path.join(dataset_dir, "split/overview.csv"), index=False)
    
        