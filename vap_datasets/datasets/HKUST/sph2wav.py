import glob
import os
import shutil
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf

from vap_datasets.utils.utils import repo_root
try:
    from sphfile import SPHFile
except ModuleNotFoundError as e:
    print("Requires sphfile `pip install sphfile` to process Callhome sph-files.")
    raise e


def sph_to_wav(sph_file, out_path):
    data, sr = sf.read(sph_file)
    sf.write(out_path, data, 8000, subtype="PCM_16")



if __name__ == "__main__":

    dataset = "HKUST"
    data_dir = os.path.join(repo_root(), "source", dataset, "sph/dev")
    audio_dir = os.path.join(repo_root(), "source", dataset, "wav")
    os.makedirs(audio_dir, exist_ok=True)
    sph_files = glob.glob(os.path.join(data_dir, "*.sph"), recursive=True)
    
    for sph_file in tqdm(sph_files):
        out_path = os.path.join(audio_dir, os.path.basename(sph_file).replace(".sph", ".wav"))
        sph_to_wav(sph_file, out_path)


    # #モノラル変換
    # paths = glob.glob(os.path.join(directly, "data/8k/*.wav"))
    
    # for filepath in tqdm(paths):
    #     x, fs = sf.read(filepath)
    #     basename = os.path.splitext(os.path.basename(filepath))[0]
    #     sf.write(os.path.join(directly, "mono_data", basename + "_L.wav"), x[:,0], fs)
    #     sf.write(os.path.join(directly, "mono_data", basename + "_R.wav"), x[:,1], fs)


