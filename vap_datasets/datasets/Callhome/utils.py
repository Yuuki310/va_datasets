import glob
import os 
from os.path import join, exists, basename
import re
import json as json
import numpy as np
import soundfile as sf

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

def load_data(audio_dir, text_dir):
    if not exists(text_dir):
        raise FileNotFoundError(f"text_path not found: {text_dir}")

    dataset = []
    for file in os.listdir(audio_dir):
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
    speak = False
    for row in read_txt(filepath):
        # omit empty rows and rows starting with '#' (global file info)
        if row == "" or row.startswith("#"):
            continue
        elif row.startswith("@"):
            continue
        elif row.startswith("%"):
            continue
        elif row[0]=="*":
            speak = True
            data.append(row)
        else:
            if speak == True:
                data[-1] += " " + row
    return data


def load_utterances(filepath, clean=True):
    #try:
    data = preprocess_utterance(filepath)

    last_speaker = None
    utterances = []
    script_start = -1
    script_end = -1
    member = 0

    for row in data:
        split = row.split(" ")
        time_stamp = re.findall(r"\x15(.\d*)_(.\d*)", row)
        if time_stamp == []:
            continue
        else:
            start, end = list(map(int, list(time_stamp[0])))
            #開始時間の調整
            if script_start == -1:
                script_start = start
            start = (start - script_start) / 1000
            end = (end - script_start)  / 1000

        speaker = re.findall(r"\*(.):", row)[0]
        re.sub(r"\*(.):", "", row)
        speaker = 0 if speaker == "A" else 1
        text = " ".join(split[1:])
        if last_speaker is None:
            utterances.append(
                {"start": start, "end": end, "speaker": speaker, "text": text}
            )
        # elif last_speaker == speaker:
        #     utterances[-1]["end"] = end
        #     utterances[-1]["text"] += " " + text
        else:
            utterances.append(
                {"start": start, "end": end, "speaker": speaker, "text": text}
            )
        last_speaker = speaker
    script_end = list(map(int, list(time_stamp[0])))[1]
    return script_start, script_end, utterances

def extract_vad(utterances, length):
    vad = [[], []]
    # print(utterances)
    for utt in utterances:
        if utt["end"] > length:
            break
        vad[utt["speaker"]].append((utt["start"], utt["end"]))
    return vad
