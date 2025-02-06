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

    dataset = "Sales"

    audio_dir = os.path.join(repo_root(), "data", dataset, "audio")
    vad_dir = os.path.join(repo_root(), "data", dataset, "vad")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(vad_dir, exist_ok=True)

    operator_paths = glob.glob(os.path.join(repo_root(), "source", dataset, "wav/*/op_snd/*.wav"))
    audio_source_dir = os.path.join(repo_root(), "source", dataset, "wav")
    count = 0
    data = []
    
    overview = []
    for i in range(10):
        num = str(i+1).zfill(2)
        op = glob.glob(os.path.join(audio_source_dir, num, "op_snd/*.wav"))       
        print(op)
        for op_path in op:
            op_basename = os.path.splitext(os.path.basename(op_path))[0]
            tmp_session = op_basename[:5]
            user_path = glob.glob(os.path.join(audio_source_dir, num, "user_snd", tmp_session + "*.wav"))[0]
            user_basename = os.path.splitext(os.path.basename(user_path))[0]

            # 01_2_op_01_05.wav
            # 01_2_user_02_05.wav
            # 01_2_2_01_02_05.json
            session = tmp_session + op_basename[8:10] + user_basename[9:16]

            l_audio, sr = sf.read(op_path)
            r_audio, _ = sf.read(user_path) 

            if len(l_audio) != len(r_audio):
                continue     
            
            audio_path = os.path.join(audio_dir, session + ".wav")
            stereo_audio = np.column_stack((l_audio, r_audio))
            sf.write(audio_path, stereo_audio, sr)

            tr_path = os.path.join(repo_root(), "source", dataset, "transcript", session + ".xlsx")
            vad = extract_vad(tr_path)        
            vad_path = os.path.join(vad_dir, session + ".json")
            encode_data = json.dumps(vad, indent=4)
            with open(vad_path, "w") as f:
                json.dump(vad, f, indent=4)
                
            overview.append({
                "session" : session
            })
    
        new_df = pd.DataFrame(overview)
        os.makedirs(os.path.join(repo_root(), "data", dataset, "split"), exist_ok=True)
        new_df.to_csv(os.path.join(repo_root(), "data", dataset, "split", "overview.csv"), index=False)


