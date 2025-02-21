from os.path import join, exists
import re
import os
import sys
import chardet
from vap_datasets.utils.utils import read_txt
import glob

def load_data(audio_dir, trans_dir):
    if not exists(trans_dir):
        raise FileNotFoundError(f"text_path not found: {trans_dir}")

    dataset = []
    for file in os.listdir(trans_dir):
        if file.endswith(".wav"):
            trans = glob.glob(join(trans_dir, "*", file.replace(".wav", ".txt")))[0]
            sample = {"audio": join(audio_dir, file)}
            if not exists(trans):
                continue
            sample["transcript"] = trans
            dataset.append(sample)
    return dataset



def load_utterances(filepath, clean=True):
    data = []
    with open(filepath, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    for row in read_txt(filepath, encoding=encoding):
        if row == "" or row.startswith("#"):
            continue
        else:
            row = row.split(" ")
            start = float(row[0])
            end = float(row[1])
            speaker_id = row[2][0]
            speaker = 0 if speaker_id == "A" else 1
            contents = " ".join(row[3:])
            if contents in ["<noise>"]:
                continue
            data.append({"start":start, "end":end, "speaker":speaker, "contents":contents})
            # print({"start":start, "end":end, "speaker":speaker, "contents":contents})
    return data


def extract_vad(filepath):
    utterances = load_utterances(filepath)

    vad = [[], []]
    for utt in utterances:
        vad[utt["speaker"]].append((utt["start"], utt["end"]))
    return vad




if __name__ == "__main__":

    from os import listdir
    from datasets_turntaking.utils import read_json

    extracted_path = "/home/z40351r/.cache/huggingface/datasets/downloads/extracted/f061a9d1adb5f0388f08608aef42a37acdfd81fcd42a22dc66de1df216324b6c"
    session = "2001"
    session = "4936"

    session = str(session)
    session_dir = join(extracted_path, "swb_ms98_transcriptions", session[:2], session)
    print(listdir(session_dir))
    dialog = extract_dialog(session, session_dir)

    print(len(dialog))
    for row in dialog[1]:
        print(row["text"])

    vad = extract_vad_list_from_words(dialog)
    print(vad)
