import pandas as pd
import json
import os
from vap.utils.utils import repo_root

def pearsonr(datalist, option=None):
    df = pd.DataFrame({})
    for dataname in datalist:
        if option == "sym":
            path = os.path.join(repo_root(), "data", dataname, "sym_labels_train.json")
        else:
            path = os.path.join(repo_root(), "data", dataname, "labels_train.json")
        json_open = open(path)
        dict = json.load(json_open)
        ls = []
        for key, value in dict.items():
            ls.append(value)
        # for i in range(256):
            # ls.append(dict[str(i)])
        df[dataname] = ls
    corr = df.corr()
    spearman = df.corr(method="spearman")
    # print(corr)
    print(spearman)
    return corr, spearman


def main():
    datalist = []
    # Callhome = ["Callhome_eng", "Callhome_deu", "Callhome_jpn", "Callhome_spa", "Callhome_zho"]
    japanese = ["CEJC_phone", "CSJ", "Callhome_jpn", "Sales", "Travels"]
    CEJC = ["CEJC_phone","CEJC_chat","CEJC_cons","CEJC_family","CEJC_friend","CEJC_works"]
    Callhome_jpn = ["Callhome_jpn", "Callhome_jpn_webrtc"]
    # pearsonr(Callhome + ["Switchboard"], option="sym")
    pearsonr(japanese + ["Callhome_jpn_webrtc"], option="sym")
    pearsonr(["Switchboard", "Switchboard_webrtc", "Switchboard_webrtc_f10", "Callhome_eng", "Callhome_eng_webrtc", "Callhome_eng_webrtc_f10", "Callhome_jpn","Callhome_jpn_webrtc"], option="sym")
    # pearsonr(["CEJC","CEJC_phone","CEJC_cons","CEJC_chat","CEJC_"], option="sym")
    # pearsonr(CEJC + Callhome_jpn, option="sym")

    # for dataset in Callhome:
    #     corr = pearsonr("Callhome_eng", dataset)
    # for dataset in Callhome:
    #     corr = pearsonr("Switchboard", dataset)
    # for dataset in japanese:
    #     corr = pearsonr("Callhome_jpn", dataset)
    # corr = pearsonr("Sales", "Travels")
    # for dataset in japanese:
    #     corr = pearsonr("CEJC_phone", dataset)

if __name__ == "__main__":
    # paths = ["/data/group1/z40351r/VAP8k/data/Callhome_eng/labels_train.json", "/data/group1/z40351r/VAP8k/data/Callhome_deu/labels_train.json"]
    main()
    

# json_open = open(path1)
# dict_1 = json.load(json_open)

# print(json_load)

# df = pd.DataFrame({'x': [-0.5,  1.1, -1.6, -1.3,  0.8,  0.9, -0.2, -0.8,  0.6,  0.9],
#                    'y': [-0.2, -1.1, -1.8, -0.3,  0.2,  1.3,  0.3, -0.3,  0.4,  1.5]})
# print(df)

