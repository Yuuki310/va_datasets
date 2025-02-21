import os
import re
import glob
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_data,
    load_utterances,
    extract_vad,
)
from datasets_turntaking.utils import (
    repo_root,
    read_txt,
    read_json,
)

dataset = "CEJC"


def prepare_df(data_dir, csv_path, participant_path):
    df = pd.read_csv(csv_path)
    participant_df = pd.read_csv(participant_path)
    sample_list = []
    count_over3sp = 0
    count_over3lb = 0
    count_no_sound = 0
    for index, row in df.iterrows():
        rec = row["協力者ID"]
        session = row["セッションID"]
        talk = row["会話ID"]
        speaker = row["話者数"]
        format = row["形式"]
        place = row["場所"]
        activity = row["活動"]
        relation = row["話者間の関係性"]
        # 話者数2に限定
        if speaker != 2: 
            count_over3sp += 1
            continue
        participants = participant_df[participant_df["会話ID"] == talk]
        # 話者数2にもかかわらず話者ラベルが3人以上いる場合があるので削除
        if len(participants) != 2: 
            count_over3lb += 1
            continue
        # 話者ラベルがNで始まる場合単独音源がないので削除
        # ループ汚いので要整理
        speaker_label = []
        for _, participant in participants.iterrows():
            if participant["話者ラベル"].startswith("N") : 
                count_no_sound += 1
                break
            speaker_label.append(participant["話者ラベル"][:4])
        else: 
            sample = {
                "rec": rec,
                "session": session,
                "talk":talk,
                "first_speaker_id":speaker_label[0], 
                "second_speaker_id":speaker_label[1], 
                "format": format,
                "place": place,
                "activity": activity,
                "relation": relation,
            }
            sample_list.append(sample)

    # print("Total number of conversations: ", len(df))
    # print("The number of speakers is stated as three or more :", count_over3sp)
    # print("The number of speakers is stated as two, but in reality, there are three or more speakers: ", count_over3lb)
    # print("Without corresponding audio sources for each speaker: ", count_no_sound)
    # print("Available for use as two-person dialogues: ", len(sample_list))

    new_df = pd.DataFrame.from_dict(sample_list)
    return new_df

if __name__ == "__main__":
    audio_dir = os.path.join(repo_root(), "data", dataset, "audio/8k")
    trans_dir = os.path.join(repo_root(), "data", dataset, "data")
    vad_dir = os.path.join(repo_root(), "data", dataset, "vad")
    os.makedirs(vad_dir, exist_ok=True)


    meta_path = os.path.join(repo_root(), "data", dataset, "metainfo/会話.csv")
    meta_df = pd.read_csv(meta_path)
    participant_path = os.path.join(repo_root(), "data", dataset, "metainfo/SUW/participant.csv")
    df = prepare_df(trans_dir, meta_path, participant_path)
    df.to_csv(os.path.join(repo_root(), "datasets_turntaking", "datasets", dataset, "files/sessions.csv"), index=False)
    data = []
    for _, row in df.iterrows():
        transcript_path = os.path.join(trans_dir, row["rec"], row["session"], row["talk"] + "-transUnit.csv")
        dialog = load_utterances(transcript_path, row["first_speaker_id"], row["second_speaker_id"])

        vad = extract_vad(dialog)
        vad_path = os.path.join(vad_dir, f"{row['talk']}.json") 
        first_audio_path = os.path.join(audio_dir, row["talk"] + "_" + row["first_speaker_id"] + ".wav")
        second_audio_path = os.path.join(audio_dir, row["talk"] + "_" + row["second_speaker_id"] + ".wav")

        encode_data = json.dumps(vad, indent=4)

        with open(vad_path, "w") as f:
            json.dump(vad, f, indent=4)

        data.append(
            {   
                "first_audio_path": str(first_audio_path),
                "second_audio_path": str(second_audio_path),
                "vad_path": vad_path,
                "format" :  row["format"],
                "place" : row["place"],
                "activity" : row["activity"],
                "relation" : row["relation"],
            }
        )

    new_df = pd.DataFrame(data)
    new_df.to_csv(os.path.join(repo_root(), "datasets_turntaking", "datasets", dataset, "files", "audio_vad.csv"), index=False)
    