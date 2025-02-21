import glob
import os
import shutil
from tqdm import tqdm



if __name__ == "__main__":

    directly = "/data/group1/z40351r/datasets_turntaking/data/CEJC2021"

    
    files = glob.glob(os.path.join(directly, "*.csv"))
    print(files)
    for filepath in tqdm(files):
        session_name = os.path.splitext(os.path.basename(filepath))[0]
        session = session_name.split("_")[0]
        section = session_name.split("_")[1][:3]
        new_dir = os.path.join(directly, "data", session, session + "_" + section)
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(filepath, new_dir)

