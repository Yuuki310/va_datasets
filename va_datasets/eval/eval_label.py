import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import json
import os
import pandas as pd
from argparse import ArgumentParser

from vap_datasets.utils.utils import repo_root
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from itertools import chain
from tqdm import tqdm 

def eval_label(pred, truth):
    pred = list(chain(*pred))
    truth = list(chain(*truth))
    # return F.binary_cross_entropy_with_logits(vad, groundtruth)
    f1 = f1_score(truth, pred)
    f1_macro = f1_score(truth, pred, average='macro')
    accuracy =  accuracy_score(truth, pred)
    precision = precision_score(truth, pred)
    recall = recall_score(truth, pred)
    return f1, f1_macro, accuracy, precision, recall

def time_to_frames(time):
    return int(time * 25)


def fill_intervals(binary, num_frames):
    vad_tensor = torch.zeros((num_frames, 2))

    for speaker in range(2):
        for v in binary[speaker]:
            s = time_to_frames(v[0])
            e = time_to_frames(v[1])
            vad_tensor[s:e,speaker] = 1
    return vad_tensor


def load_label(label_path, groundtruth_path):
    with open(label_path, 'r', encoding='utf-8') as file:
        label_file = json.load(file)
    with open(groundtruth_path, 'r', encoding='utf-8') as file:
        groundtruth_file = json.load(file)
    num_frames = 25 * max([len(label_file[0]), len(groundtruth_file[0])])

    label_binary_list = fill_intervals(label_file, num_frames)
    groundtruth_binary_list = fill_intervals(groundtruth_file, num_frames)

    return label_binary_list, groundtruth_binary_list


def main(args):
    dataset = args.dataset
    label_name = args.label_name
    groundtruth_name = args.groundtruth_name
    split = args.split
    dataset_dir = os.path.join(repo_root(), "data", dataset)
    label_dir = os.path.join(dataset_dir, "labels", label_name)
    groundtruth_dir = os.path.join(dataset_dir, "labels", groundtruth_name)
    df = pd.read_csv(os.path.join(dataset_dir, f"split/{split}.csv"))

    result_list = []
    f1_scores, f1_macro_scores, accuracies, precisions, recalls = [], [], [], [], []

    for index, row in tqdm(df.iterrows()):
        session = str(row["session"]).zfill(4)
        label_path = os.path.join(label_dir, session + ".json")
        groundtruth_path = os.path.join(groundtruth_dir, session + ".json")
        label, groundtruth = load_label(label_path, groundtruth_path)

        f1score, f1_macro, accuracy, precision, recall = eval_label(label, groundtruth)

        f1_scores.append(f1score)
        f1_macro_scores.append(f1_macro)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        # result_list.append(loss)
        # with open(output_path, "w") as f:
        #     json.dump(out_vad, f, indent=4) 

    avg_f1_score = np.mean(f1_scores)
    avg_f1_macro_score = np.mean(f1_macro_scores)
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    data = {
        "f1_macro_score": avg_f1_macro_score,
        "f1_score": avg_f1_score,
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall
    }
    output_path = os.path.join(label_dir, f".eval-{split}.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4) 

    # 出力
    print(f"Average F1 macro Score: {avg_f1_macro_score}")
    print(f"Average F1 Score: {avg_f1_score}")
    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--label_name", type=str)
    parser.add_argument("--groundtruth_name", type=str, default="original")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    main(args)