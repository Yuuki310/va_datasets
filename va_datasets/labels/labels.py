# coding: UTF-8
import json
import os
import warnings
from os.path import join, dirname, basename, exists
from pprint import pprint

import warnings
from hydra.utils import instantiate
import torch
import json
from omegaconf import DictConfig, OmegaConf 
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
import numpy as np
import hydra
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from vap.modules.lightning_module import VAPModule
from vap.plot.plot_utils import plot_stereo
from vap.objective import VAPObjective
from vap_datasets.utils.utils import repo_root
from collections import Counter
warnings.simplefilter("ignore")

class Labels:
    def __init__(self, conf, dm, model_path=None, find_threshold = False):
        self.conf = conf
        self.dataset = self.conf["datamodule"]["datasets"][0]
        self.subsets = self.conf["datamodule"]["subsets"]

        self.dm = dm
        if model_path != None:
            self.model_path = model_path
            self.model = VAPModule.load_model(model_path).to("cuda")
            self.model = self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.result = None
        self.turn_taking_probs = None
        self.probs = None
        self.events = None
        self.train_counter = None
        self.test_counter = None


    def logit_class(self):
        assert self.model_path != None, "'model_path' is undefined."
        test_dataset = self.conf["datamodule"]["datasets"][0]

        indexlist = []
        sum_probs = torch.zeros(256).to("cuda")
        for ii, batch in tqdm(
            enumerate(self.dm.test_dataloader()),
            total=len(self.dm.test_dataloader()),
            dynamic_ncols=True,
            leave=False,
        ):  
            waveform = batch["waveform"].to("cuda")
            out = self.model.probs(waveform)
            
            # index mode
            for probs in out["probs"][0]:
                index = torch.argmax(probs)
                indexlist.append(int(index))
            
            # probs mode
            # 時間軸方向にforを回す
            for probs in out["probs"][0]:
                sum_probs += probs
        print(sum_probs)
            
    
        counter = Counter(indexlist)
        counter_dict = {}
        probs_dict = {}
        for num in range(256):
            counter_dict[num] = counter[num]
            probs_dict[num] = int(sum_probs[num])
        
        # counter
        json_dir_path = os.path.join(os.path.dirname(self.model_path), "labels", "json")
        json_path = os.path.join(json_dir_path, test_dataset + ".json")
        os.makedirs(json_dir_path, exist_ok=True)
        with open(json_path, mode="wt") as f:
            json.dump(counter_dict, f, indent=4)

        img_dir_path = os.path.join(os.path.dirname(self.model_path), "labels", "img")
        img_path = os.path.join(img_dir_path, test_dataset + ".png")      
        os.makedirs(img_dir_path, exist_ok=True)  
        x = [i for i in range(256)]
        plt.bar(x, counter_dict.values())
        plt.savefig(img_path)
        
        plt.clf() 
        
        #probs
        json_dir_path = os.path.join(os.path.dirname(self.model_path), "probs", "json")
        json_path = os.path.join(json_dir_path, test_dataset + ".json")
        os.makedirs(json_dir_path, exist_ok=True)
        with open(json_path, mode="wt") as f:
            json.dump(probs_dict, f, indent=4)

        img_dir_path = os.path.join(os.path.dirname(self.model_path), "probs", "img")
        img_path = os.path.join(img_dir_path, test_dataset + ".png")      
        os.makedirs(img_dir_path, exist_ok=True)  
        x = [i for i in range(256)]
        plt.bar(x, probs_dict.values())
        plt.savefig(img_path)
        
        
    def labels_class(self):
        test_dataset = self.conf["datamodule"]["datasets"][0]

        ob = VAPObjective()
        label_counter = Counter()
        for ii, batch in tqdm(
            enumerate(self.dm.test_dataloader()),
            total=len(self.dm.test_dataloader()),
            dynamic_ncols=True,
            leave=False,
        ):  
            vad_label = batch["vad"].to("cuda")[0].cpu()
            vap_label = ob.get_labels(vad_label)
            label_counter.update(vap_label.tolist())

        counter_dict = {}
        for num in range(256):
            counter_dict[num] = label_counter[num]

        # counter
        class_dir = os.path.join(repo_root(), "data", test_dataset, "class", self.subsets[0])
        os.makedirs(class_dir, exist_ok=True)
        json_path = os.path.join(class_dir, "labels_test.json")
        with open(json_path, mode="wt") as f:
            json.dump(counter_dict, f, indent=4)
        self.test_counter = counter_dict

        img_path = os.path.join(class_dir, "labels_test.png")
        x = [i for i in range(256)]
        plt.clf() 
        plt.bar(x, counter_dict.values())
        plt.savefig(img_path)
    
        # ここからtrainで同じことやる
        # 要修正
        
        ob = VAPObjective()
        label_counter = Counter()
        for ii, batch in tqdm(
            enumerate(self.dm.train_dataloader()),
            total=len(self.dm.train_dataloader()),
            dynamic_ncols=True,
            leave=False,
        ):  
            vad_label = batch["vad"].to("cuda")[0].cpu()
            vap_label = ob.get_labels(vad_label)
            label_counter.update(vap_label.tolist())

        counter_dict = {}
        for num in range(256):
            counter_dict[num] = label_counter[num]

        # counter
        json_path = os.path.join(class_dir, "labels_train.json")
        with open(json_path, mode="wt") as f:
            json.dump(counter_dict, f, indent=4)
        self.train_counter = counter_dict

        img_path = os.path.join(class_dir, "labels_train.png")
        x = [i for i in range(256)]
        plt.clf() 

        plt.bar(x, counter_dict.values())
        plt.savefig(img_path)            
        print(img_path)   
        
    def to_device(batch, device="cuda"):
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v.to(device)
            else:
                new_batch[k] = v
      
    # def load_json(self):
    #     df = pd.DataFrame({})
    #     path = os.path.join(repo_root(), "data", dataname, "labels_train.json")
    #     json_open = open(path)
    #     dict = json.load(json_open)
        
    def symmetry(self):
        dataset = self.conf["datamodule"]["datasets"][0]

        ls = []
        for i in range(256):
            ls.append(self.train_counter[int(i)])

        new_dict = {}
        processed = []
        num = 0
        for i in range(256):
            b = int(i / 16)
            c = int(i % 16) 
            pair_idx = 16 * c + b

            if pair_idx in processed:
                continue
            else:
                if i == pair_idx:
                    new_dict[num] = ls[i]
                else:
                    new_dict[num] = ls[i] + ls[pair_idx]
                processed.append(i)
                num += 1
            
        class_dir = os.path.join(repo_root(), "data", dataset, "class", self.subsets[0])
        json_path = os.path.join(class_dir, "sym_labels_train.json")
        with open(json_path, mode="wt") as f:
            json.dump(new_dict, f, indent=4)

        img_path = os.path.join(class_dir, "sym_labels_train.png")
        x, y = zip(*sorted(new_dict.items()))
        plt.clf() 
        plt.bar(x, new_dict.values())
        plt.savefig(img_path)
        


@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    # model_path = os.path.join(cfg.exp_dir, "best_model.ckpt")    
    cfg.datamodule.batch_size = 1
    cfg_dict = dict(OmegaConf.to_object(cfg))
    dm = instantiate(cfg.datamodule)
    dm.setup("test")
    dm.setup()

    labels = Labels(cfg_dict, dm)
    # labels.logit_class()
    labels.labels_class()
    labels.symmetry()


if __name__ == "__main__":
    # args = get_args()
    # print(args)
    main()
