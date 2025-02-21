from os.path import exists
import pandas as pd
import json
from typing import Optional, Mapping
import matplotlib.pyplot as plt
import os
from itertools import chain
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import lightning as L

from vap.utils.audio import load_waveform, mono_to_stereo
from vap.utils.utils import vad_list_to_onehot
from vap.utils.plot import plot_melspectrogram, plot_vad

from vap_datasets.utils.utils import repo_root

SAMPLE = Mapping[str, Tensor]


def load_df(path: str) -> pd.DataFrame:
    def _vl(x):
        return json.loads(x)

    def _session(x):
        return str(x)

    converters = {
        "vad_list": _vl,
        "session": _session,
    }
    return pd.read_csv(path, converters=converters)


def load_mono_df(path: str) -> pd.DataFrame:
    def _vl(x):
        return json.loads(x)

    def _session(x):
        return str(x)

    converters = {
        "vad_list": _vl,
        "session": _session,
    }
    df = pd.read_csv(path, converters=converters)
    counter_df = df.copy()
    df["speaker"] = 0
    counter_df["speaker"] = 1
    tmp = []
    for index, row in counter_df.iterrows():
        counter_df.at[index, "vad_list"] = [row.vad_list[1], row.vad_list[0]]

    new_df = pd.concat([df,counter_df], ignore_index=True)
    return new_df


def force_correct_nsamples(w: Tensor, n_samples: int) -> Tensor:
    if w.shape[-1] > n_samples:
        w = w[:, -n_samples:]
    elif w.shape[-1] < n_samples:
        w = torch.cat([w, torch.zeros_like(w)[:, : n_samples - w.shape[-1]]], dim=-1)
    return w


def plot_dset_sample(d):
    """
    VAD is by default longer than the audio (prediction horizon)
    So you will see zeros at the end where the VAD is defined but the audio not.
    """
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].set_title(d["session"])
    plot_melspectrogram(d["waveform"], ax=ax[:2])
    x = torch.arange(d["vad"].shape[0]) / dset.frame_hz
    plot_vad(x, d["vad"][:, 0], ax[0], ypad=2, label="VAD 0")
    plot_vad(x, d["vad"][:, 1], ax[1], ypad=2, label="VAD 1")
    _ = [a.legend() for a in ax]
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


class VAPDataset(Dataset):
    def __init__(
        self,
        path: str,
        horizon: float = 2,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        merge: bool = False,
        mono: bool = False,
        mode: str = "stereo",
    ) -> None:
        # mode : {stereo, mono, onside}
        # mono: LRどちらかのモノラル音源を出力
        # oneside: Lは残してRを無音とするステレオ音源を出力
        self.path = path
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.horizon = horizon
        self.merge = merge
        self.mono = mono
        self.mode = mode

        if self.mode == "stereo" :
            self.df = load_df(path)
        else:
            self.df = load_mono_df(path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> SAMPLE:
        d = self.df.iloc[idx]
        # Duration can be 19.99999999999997 for some clips and result in wrong vad-shape
        # so we round it to nearest second

        dur = round(d["end"] - d["start"])
        w, _ = load_waveform(
            audio_path=d["audio_path"],
            start_time=d["start"],
            end_time=d["end"],
            sample_rate=self.sample_rate,
            mono=self.mono,
        )
        # Ensure correct duration
        # Some clips (20s) becomes
        # [2, 320002] insted of [2, 320000]
        # breaking the batching
        n_samples = int(dur * self.sample_rate)
        w = force_correct_nsamples(w, n_samples)
        # Stereo Audio
        # Use the vad-list information to convert mono to stereo
        if w.shape[0] == 1:
            w = mono_to_stereo(w, d["vad_list"], sample_rate=self.sample_rate)
        if w.shape[0] == 4:
            print(f'{d["audio_path"]} : stereo size error')
        vad = vad_list_to_onehot(
            d["vad_list"], duration=dur + self.horizon, frame_hz=self.frame_hz
        )
        items = {
            "session": d.get("session", ""),
            "vad": vad,
            "dataset": d.get("dataset", ""),
        }
        if self.mode == "mono":
            w0 = w[d["speaker"]].unsqueeze(dim=0)
            w1 = w[1 - d["speaker"]].unsqueeze(dim=0)    
            w0 = force_correct_nsamples(w0, n_samples)
            w1 = force_correct_nsamples(w1, n_samples)
            items["waveform"] = w0
            items["counter_waveform"] = w1            
        elif self.mode == "oneside": 
            w0 = w[d["speaker"]]
            w0 = force_correct_nsamples(w0, n_samples)
            silent = torch.zeros_like(w0)
            stereo_waveform = torch.stack([w0, silent], dim=0)
            items["waveform"] = stereo_waveform
        else:
            items["waveform"] = w
        return items



class VAPDataModule(L.LightningDataModule):
    def __init__(
        self,
        datasets: list[str] = None,
        subsets: list[str] = ["default"],
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        quantity: int = None, 
        quantities: list[int] = None,
        horizon: float = 2,
        sample_rate: int = 16000,
        frame_hz: int = 50,
        mono: bool = False,
        batch_size: int = 4,
        num_workers: int = 2,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        mode: str = "stereo",
        rand: bool = False,
        **kwargs,
    ):
        super().__init__()
        print("datasets:",datasets)
        print("subset:",subsets)
        print("quantity:",quantity)
        if datasets != None:
            print(datasets)
            if len(datasets) == 1:
                subset = subsets[0]
                data_path = os.path.join(repo_root(), "data", datasets[0], "path", subset)
                train_path = os.path.join(data_path, "sliding_window_train.csv")
                test_path = os.path.join(data_path, "sliding_window_test.csv")
                val_path = os.path.join(data_path, "sliding_window_val.csv")              
                train_path = train_path if os.path.exists(train_path) else None
                test_path = test_path if os.path.exists(test_path) else None
                val_path = val_path if os.path.exists(val_path) else None 

                if quantity != None:
                    # quantity個のデータのリストをcacheに保存（train）
                    quantity = int(quantity)
                    set_name = os.path.basename(data_path) + str(quantity)
                    cache_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), ".cache", set_name, subset)
                    os.makedirs(cache_dir, exist_ok =True)
                    if rand == False:
                        train_df = pd.read_csv(train_path).iloc[:quantity]
                    else:
                        train_df = pd.read_csv(train_path).sample(n=quantity)
                    
                    train_path = os.path.join(cache_dir, "sliding_window_train.csv") 
                    train_df.to_csv(train_path)

                    # quantity個のデータのリストをcacheに保存（test）
                    # t-sne用
                    quantity = int(quantity)
                    set_name = os.path.basename(data_path) + str(quantity)
                    cache_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), ".cache", set_name, subset)
                    os.makedirs(cache_dir, exist_ok =True)
                    test_df = pd.read_csv(test_path)
                    if len(test_df) > quantity:
                        test_df = test_df.sample(n=quantity)
                    
                    test_path = os.path.join(cache_dir, "sliding_window_test.csv") 
                    test_df.to_csv(test_path)

            else:

                train_path_list = []
                test_path_list = []
                val_path_list = []
                for i, data_path in enumerate(datasets):
                    if len(subsets) == 1:
                        subset = subsets[0]
                    else:
                        subset = subsets[i]
                    data_path = os.path.join(repo_root(), "data", data_path, "path", subset)
                    train_path = os.path.join(data_path, "sliding_window_train.csv")
                    test_path = os.path.join(data_path, "sliding_window_test.csv")
                    val_path = os.path.join(data_path, "sliding_window_val.csv")
                    if os.path.exists(train_path):
                        train_path_list.append(train_path)
                    if os.path.exists(test_path):
                        test_path_list.append(test_path)
                    if os.path.exists(val_path):
                        val_path_list.append(val_path)
                set_name = "+".join([os.path.basename(data_path) + str(quantities[i]) for i, data_path in enumerate(datasets)])
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(data_path))), ".cache", set_name)
                os.makedirs(cache_dir, exist_ok =True)
                
                # trainデータの処理
                if quantities == None:
                    quantities = [len(pd.read_csv(path)) for path in train_path_list]
                df = pd.DataFrame()
                for i, path in enumerate(train_path_list):
                    quantity = quantities[i]
                    print(path, quantity)

                    df = pd.concat([df, pd.read_csv(path).sample(n=quantity)], ignore_index=True)
                if train_path_list != []:
                    train_path = os.path.join(cache_dir, "sliding_window_train.csv") 
                    df.to_csv(train_path)

                # 評価データの処理
                min_size = min([len(pd.read_csv(path)) for path in test_path_list])
                df = pd.DataFrame()
                for i, path in enumerate(test_path_list):
                    df = pd.concat([df, pd.read_csv(path).sample(n=min_size)], ignore_index=True)
                if test_path_list != []:
                    test_path = os.path.join(cache_dir, "sliding_window_test.csv") 
                    df.to_csv(test_path)

                # valデータの処理
                # quantityを適用せず最小のデータセットに合わせる
                min_size = min([len(pd.read_csv(path)) for path in val_path_list])
                df = pd.DataFrame()
                for i, path in enumerate(val_path_list):
                    df = pd.concat([df, pd.read_csv(path).sample(n=min_size)], ignore_index=True)
                if val_path_list != []:
                    val_path = os.path.join(cache_dir, "sliding_window_val.csv") 
                    df.to_csv(val_path)

                    
        # Files
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # values
        self.mono = mono
        self.mode = mode
        self.horizon = horizon
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz

        # DataLoder
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tTrain: {self.train_path}"
        s += f"\n\tVal: {self.val_path}"
        s += f"\n\tTest: {self.test_path}"
        s += f"\n\tHorizon: {self.horizon}"
        s += f"\n\tSample rate: {self.sample_rate}"
        s += f"\n\tFrame Hz: {self.frame_hz}"
        s += f"\nData"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"
        s += f"\n\tprefetch_factor: {self.prefetch_factor}"
        return s

    def prepare_data(self):
        if self.train_path is not None:
            assert self.path_exists("train"), f"No TRAIN file found: {self.train_path}"

        if self.val_path is not None:
            assert self.path_exists("val"), f"No TRAIN file found: {self.train_path}"

        if self.test_path is not None:
            assert exists(self.test_path), f"No TEST file found: {self.test_path}"

    def path_exists(self, split):
        path = getattr(self, f"{split}_path")
        if path is None:
            return False

        if not exists(path):
            return False
        return True

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""

        if stage in (None, "fit"):
            assert self.path_exists("train"), f"Train path not found: {self.train_path}"
            assert self.path_exists("val"), f"Val path not found: {self.val_path}"
            self.train_dset = VAPDataset(
                self.train_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
                mode=self.mode,
            )
            self.val_dset = VAPDataset(
                self.val_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
                mode=self.mode,
            )

        if stage in (None, "test"):
            assert self.path_exists("test"), f"Test path not found: {self.test_path}"
            self.test_dset = VAPDataset(
                self.test_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
                mode=self.mode,
            )
        if stage in ("val"):
            assert self.path_exists("val"), f"Val path not found: {self.val_path}"
            self.val_dset = VAPDataset(
                self.val_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
                mode=self.mode,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )


if __name__ == "__main__":

    from argparse import ArgumentParser
    import torch
    from tqdm import tqdm

    # parser = ArgumentParser()
    # parser.add_argument("--csv_path", type=str, default="/data/group1/z40351r/StereoVAP/data/CEJC2021/sliding_window_train.csv")
    # parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--prefetch_factor", type=int, default=None)
    # parser.add_argument("--single", action="store_true")
    # args = parser.parse_args()

    # if args.single:
    #     dset = VAPDataset(path=args.csv_path)
    #     idx = int(torch.randint(0, len(dset), (1,)).item())
    #     d = dset[idx]
    #     plot_dset_sample(d)
    # else:

    # dm = VAPDataModule(
    #     # train_path="data/splits/sliding_window_dset_train.csv",
    #     # val_path="data/splits/sliding_window_dset_val.csv",
    #     test_path=args.csv_path,
    #     batch_size=args.batch_size,  # as much as fit on gpu with model and max cpu cores
    #     num_workers=args.num_workers,  # number cpu cores
    #     prefetch_factor=args.prefetch_factor,  # how many batches to prefetch
    #     mono = False,
    #     mode = "stereo",
    # )
    dm = VAPDataModule(
        datasets=["CallfullJPN"],
        subsets=["vadCEJCbert50-v2"],
        mono = False,
        mode = "stereo",
        sample_rate=8000,
    )
    dm.prepare_data()
    dm.setup()
    # print(dm)
    print("VAPDataModule: ", len(dm.train_dset))
    dloader = dm.val_dataloader()
    for batch in tqdm(
        dloader, desc="Running VAPDatamodule (Training wont be faster than this...)"
    ):
        print(batch["session"])
