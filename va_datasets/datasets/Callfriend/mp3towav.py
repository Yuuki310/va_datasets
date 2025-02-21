import pydub
import glob
import os
from vap_datasets.utils.utils import repo_root

files = glob.glob(os.path.join(repo_root(), "source/CallfriendENG/mp3/*"))

wav_dir_path = os.path.join(repo_root(), "source/CallfriendENG/wav")
os.makedirs(wav_dir_path, exist_ok=True)

for file in files:
    print(file)
    # files               
    dirname = os.path.dirname(file)
    basename = os.path.splitext(os.path.basename(file))[0]
    wav = os.path.join(wav_dir_path, f"{basename}.wav")
    
    # convert wav to mp3                                                            
    audio = pydub.AudioSegment.from_mp3(file)
    audio.export(wav, format="wav")