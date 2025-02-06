import re
from vap_datasets.utils.utils import read_txt
import datetime
import os
from os.path import join, exists
import glob

def load_data(audio_dir, trans_dir):
    if not os.path.exists(trans_dir):
        raise FileNotFoundError(f"text_path not found: {trans_dir}")

    dataset = []
    file_list = glob.glob(os.path.join(audio_dir, "**/D*.wav"))
    for file in file_list:
        sample = {"audio": join(audio_dir, file)}

        basename = os.path.basename(file)
        trans = glob.glob(join(trans_dir, "**", basename.replace(".wav", ".trn")))
        if len(trans) == 1 and os.path.exists(trans[0]):
            sample["transcript"] = trans[0]
            dataset.append(sample)

    return dataset


def load_utterances(filepath, clean=True):
    data = read_txt(filepath, encoding="shift_jis")
    
    utterances = []
    for row in data:
        row = row.split(" ", 2)
        start, end = row[1].split("-")
        start = float(start)
        end = float(end)
        speaker = row[2][0]
        if speaker == "L":
            speaker = 0
        elif speaker == "R":
            speaker = 1   
        text = row[2][2:]

        utterances.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )

    return utterances


def extract_vad(filepath):
    utterances = load_utterances(filepath)

    vad = [[], []]
    for utt in utterances:
        vad[utt["speaker"]].append((utt["start"], utt["end"]))
    return vad



if __name__ == "__main__":

    filepath = "/data/group1/z40351r/datasets_turntaking/data/CSJ/TRN/Form2/noncore/D03M0013.trn"
    utterances = load_utterances(filepath)
    print(utterances)

    vad = extract_vad(utterances)
    print(vad)
    