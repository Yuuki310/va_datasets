import glob
import os 
from os.path import join, exists, basename
import re
import json as json
import numpy as np
import soundfile as sf
import pandas as pd
from datasets_turntaking.utils import read_txt

def callhome_regexp(s):
    """callhome specific regexp"""
    # {text}:  sound made by the talker {laugh} {cough} {sneeze} {breath}
    s = re.sub(r"\{.*\}", "", s)

    # [[text]]: comment; most often used to describe unusual
    s = re.sub(r"\[\[.*\]\]", "", s)

    # [text]: sound not made by the talker (background or channel) [distortion]    [background noise]      [buzz]
    s = re.sub(r"\[.*\]", "", s)

    # (( )): unintelligible; can't even guess text
    s = re.sub(r"\(\(\s*\)\)", "", s)

    # single word ((taiwa))
    # ((text)): unintelligible; text is best guess at transcription ((coffee klatch))
    s = re.sub(r"\(\((\w*)\)\)", r"\1", s)

    # multi word ((taiwa and other))
    # s = re.sub(r"\(\((\w*)\)\)", r"\1", s)
    s = re.sub(r"\(\(", "", s)
    s = re.sub(r"\)\)", "", s)

    # -text / text-: partial word = "-tion absolu-"
    s = re.sub(r"\-(\w+)", r"\1", s)
    s = re.sub(r"(\w+)\-", r"\1", s)

    # +text+: mispronounced word (spell it in usual orthography) +probably+
    s = re.sub(r"\+(\w*)\+", r"\1", s)

    # **text**: idiosyncratic word, not in common use
    s = re.sub(r"\*\*(\w*)\*\*", r"\1", s)

    # remove proper names symbol
    s = re.sub(r"\&(\w+)", r"\1", s)

    # remove non-lexemes symbol
    s = re.sub(r"\%(\w+)", r"\1", s)

    # text --             marks end of interrupted turn and continuation
    # -- text             of same turn after interruption, e.g.
    s = re.sub(r"\-\-", "", s)

    # <language text>: speech in another language
    s = re.sub(r"\<\w*\s(\w*\s*\w*)\>", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    s = re.sub(r"^\s", "", s)
    return s

def load_data(data_dir):
    if not exists(data_dir):
        raise FileNotFoundError(f"text_path not found: {text_dir}")

    dataset = []
    for rec in os.listdir(data_dir):
        for session in os.listdir(rec):
            files = os.listdir(session)
            sample = {"first_audio_path": join(audio_dir, file),
                      "second_audio_path": join(audio_dir, file)}
            
        if file.endswith(".wav"):
            sample = {"audio_path": join(audio_dir, file)}
            txt = join(text_dir, file.replace(".wav", ".cha"))
            if exists(txt):
                sample["text"] = txt
            dataset.append(sample)
    return dataset


def get_speaker_num(filepath):
    num = 0
    for row in read_txt(filepath):
        if row.startswith("@ID"):
            num += 1
        elif row[0]=="*":
            break
    return num

def preprocess_utterance(filepath):
    """
    Load filepath and preprocess the annotations

    * Omit empty rows
    * join utterances spanning multiple lines
    """
    data = []
    print(filepath)
    speak = False
    df = pd.read_csv(filepath, encoding="shift_jis")
    return df


def load_utterances(filepath, first_speaker_id, second_speaker_id, clean=True):
    #try:
    df = preprocess_utterance(filepath)
    print(df)
    last_speaker = None
    utterances = []
    script_start = -1
    script_end = -1
    member = 0

    for _, row in df.iterrows():
        start = row["startTime"]
        end = row["endTime"]
        speaker = row["speakerID"][:4]
        text = row["text"]

        if speaker == first_speaker_id : speaker = 0 
        elif speaker == second_speaker_id : speaker = 1

        utterances.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )
    return utterances

def extract_vad(utterances):
    vad = [[], []]
    for utt in utterances:
        print(utt)
        vad[utt["speaker"]].append((utt["start"], utt["end"]))
    return vad
