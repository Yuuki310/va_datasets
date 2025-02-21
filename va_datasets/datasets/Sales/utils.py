import re
from vap_datasets.utils.utils import read_txt
import datetime
import pandas as pd
import openpyxl






def load_utterances(filepath, clean=True):
    utterances = []
    df = openpyxl.load_workbook(filepath)["接客"]
    for row in df.iter_rows():
        
        row = list(row)
        
        start = row[0].value
        end = row[1].value
        operator = row[2].value
        user = row[3].value

        if start == ":.": break
        if operator == None or operator == "":
            speaker = 1
            text = user
        elif user == None or user == "":
            speaker = 0
            text = operator
        else:
            continue
        start = time2sec(start)
        end = time2sec(end)
        print(start)
        utterances.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )
    print(utterances)
    return utterances

def time2sec(time):
    # time : mm:ss.ss
    print(time)
    minutes = int(time[0:2])
    seconds = int(time[3:5])
    try:
        milliseconds = int(time[6:8])*10
    except:
        milliseconds = 0
    dt = datetime.timedelta(minutes=minutes, seconds=seconds, milliseconds=milliseconds)
    total_seconds = dt.total_seconds()
    return total_seconds


def extract_vad(path):
    utterances = load_utterances(path)
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
    