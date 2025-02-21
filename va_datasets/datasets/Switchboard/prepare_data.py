import glob
import os
import shutil
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf

import librosa
try:
    from sphfile import SPHFile
except ModuleNotFoundError as e:
    print("Requires sphfile `pip install sphfile` to process Callhome sph-files.")
    raise e


def sph_to_wav(filepath):
    dirpath = "/data/group1/z40351r/datasets_turntaking/data/Switchboard/data/8k"
    wavpath = os.path.join(dirpath, os.path.basename(filepath).replace(".sph", ".wav"))
    #sph = SPHFile(filepath)
    # write out a wav file with content from 111.29 to 123.57 seconds
    # sph.write_wav(wavpath, start=111.29, end=123.57)
    #return sph.write_wav(wavpath, start=None, stop=None)
    data, sr = sf.read(filepath)
    sf.write(wavpath, data, 8000, subtype="PCM_16")
    # x, sr = librosa.load(wavpath, mono=False)
    # data = librosa.resample(x, orig_sr=sr, target_sr=16000)
    # sf.write(wavpath, data.T, 16000, subtype="PCM_16")



if __name__ == "__main__":

    directly = "/data/group1/z40351r/datasets_turntaking/data/Switchboard"

    original_data = os.path.join(directly, "original")
    sph_files = glob.glob(os.path.join(original_data, "**/*.sph"), recursive=True)
    
    for filepath in tqdm(sph_files):
        sph_to_wav(filepath)


    # #モノラル変換
    # paths = glob.glob(os.path.join(directly, "data/8k/*.wav"))
    
    # for filepath in tqdm(paths):
    #     x, fs = sf.read(filepath)
    #     basename = os.path.splitext(os.path.basename(filepath))[0]
    #     sf.write(os.path.join(directly, "mono_data", basename + "_L.wav"), x[:,0], fs)
    #     sf.write(os.path.join(directly, "mono_data", basename + "_R.wav"), x[:,1], fs)


