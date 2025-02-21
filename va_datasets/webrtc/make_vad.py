import soundfile as sf
import os
import json
import webrtc_system
import glob
from tqdm import tqdm
import pandas as pd
from vap_datasets.utils.utils import repo_root

def vad_conversion(df):
    """
    Args:
        webrtcから得られるsegmentsをfor_dfに変換したもの
    Returns:
        リスト形式のvad(mono)
    """
    vad_list = []
    current_time = 0
    for segment in df:
        vad_label = segment["vad"]
        duration = segment["duration_sec"]
        print(duration, current_time)
        if vad_label == 1:
            vad_list.append([current_time, current_time + duration])
        current_time += duration
    return vad_list
    
def make_vad(
        st_path, 
        vad_path, 
        tmp_dir : str = "/data/group1/z40351r/vap_datasets/vap_datasets/webrtc/tmp", 
        ag : int = 3
    ):
    data, sr = sf.read(st_path)

    l_channel = data[:, 0]
    r_channel = data[:, 1]

    l_path = os.path.join(tmp_dir, os.path.splitext(os.path.basename(st_path))[0] + "_L.wav")
    r_path = os.path.join(tmp_dir, os.path.splitext(os.path.basename(st_path))[0] + "_R.wav")
    
    sf.write(l_path, l_channel, sr)
    sf.write(r_path, r_channel, sr)

    print(l_path)
    l_df = webrtc_system.pipeline(l_path, tmp_dir, aggresive=ag)
    r_df = webrtc_system.pipeline(r_path, tmp_dir, aggresive=ag)
    
    l_vad = vad_conversion(l_df)
    r_vad = vad_conversion(r_df)
    vad = [l_vad, r_vad]

    os.remove(l_path)
    os.remove(r_path)
    
    with open(vad_path, "w") as f:
        json.dump(vad, f, indent=4)
        
        
def main(args):
    
    dataset = args.dataset
    paths = glob.glob(os.path.join(repo_root(), "data", dataset, "audio", "*.wav"))
    vad_dir = os.path.join(repo_root(), "data", dataset, args.out_dir)
    os.makedirs(vad_dir, exist_ok=True)
    for path in tqdm(paths):
        vad_path = os.path.join(vad_dir, os.path.splitext(os.path.basename(path))[0] + ".json")
        make_vad(path, vad_path)

    
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--out_dir", type=str)
    
    args = parser.parse_args()
    main(args)
