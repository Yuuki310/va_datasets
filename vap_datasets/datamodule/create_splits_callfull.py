from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from vap_datasets.datamodule.datamodule import load_df
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/audio_vad.csv")
    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    cut_split_dir = "/data/group1/z40351r/vap_datasets/data/CallhomeSPA/split"
    full_split_dir = "/data/group1/z40351r/vap_datasets/data/CallfullSPA/split"
    output_dir = full_split_dir
    
    # cutのほうをいったん読み込む
    train_df = load_df(os.path.join(cut_split_dir, "train.csv"))
    val_df = load_df(os.path.join(cut_split_dir, "val.csv"))
    test_df = load_df(os.path.join(cut_split_dir, "test.csv"))
    
    cut_df = load_df(os.path.join(cut_split_dir, "overview.csv"))
    full_df = load_df(os.path.join(full_split_dir, "overview.csv"))

    # 共通レコードを取り除く
    values_to_remove = cut_df["session"].tolist()
    add_df = full_df[~full_df["session"].isin(values_to_remove)]

    # 追加分を分ける
    N = len(add_df)
    train_size = int(N * args.train_size)
    val_size = int(N * args.val_size)
    add_train_df = add_df.sample(n=train_size, random_state=0)
    add_val_df = add_df.drop(add_train_df.index)

    train_df = pd.concat([train_df, add_train_df], ignore_index=True)
    val_df = pd.concat([val_df, add_val_df], ignore_index=True)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    # Save splits
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
