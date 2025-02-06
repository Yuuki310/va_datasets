import json
import torch
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from vap_datasets.utils.utils import (
    repo_root,
    load_waveform
)

def plot_compare(
    waveform: torch.Tensor,
    va_list: list[torch.Tensor],
    sample_rate: int = 8000, 
    frame_hz :int = 50,
    figsize=(9, 6),
    start=0,
    end=20,
):
    # print(va_list.size())
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    plot_waveform(waveform=waveform[0], ax=ax[0], color="blue", sample_rate=sample_rate)
    plot_waveform(waveform=waveform[1], ax=ax[0], color="orange", sample_rate=sample_rate)

    plot_waveform(waveform=waveform[0], ax=ax[1], color="blue", sample_rate=sample_rate)
    plot_waveform(waveform=waveform[1], ax=ax[1], color="orange", sample_rate=sample_rate)

    plot_va(va=va_list[0][0], ax=ax[0], color="lightblue", vad_frame=frame_hz)
    plot_va(va=va_list[0][1], ax=ax[0], color="gold", vad_frame=frame_hz)
    ax[0].set_ylabel("Original Label", fontsize=40)


    plot_va(va=va_list[1][0], ax=ax[1], color="lightblue", vad_frame=frame_hz)
    plot_va(va=va_list[1][1], ax=ax[1], color="gold", vad_frame=frame_hz)
    ax[1].set_ylabel("WebRTC VAD Label", fontsize=40)


    ax[0].set_xticks([])
    plt.subplots_adjust(
        left=0.08, bottom=None, right=None, top=None, wspace=None, hspace=0.04
    )
    
    ax[0].set_xlim([start, end])
    ax[1].set_xlim([start, end])

    return fig, ax
    
    
def plot_stereo(
    waveform: torch.Tensor,
    va: torch.Tensor = None,
    sample_rate: int = 8000, 
    frame_hz :int = 50,
    figsize=(9, 6),
):

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    plot_waveform(waveform=waveform[0], ax=ax[0], color="blue", sample_rate=sample_rate)
    plot_waveform(waveform=waveform[1], ax=ax[1], color="orange", sample_rate=sample_rate)

    plot_va(va=va[0], ax=ax[0], color="lightblue", vad_frame=frame_hz)
    plot_va(va=va[1], ax=ax[1], color="gold", vad_frame=frame_hz)

    ax[0].set_xticks([])
    plt.subplots_adjust(
        left=0.08, bottom=None, right=None, top=None, wspace=None, hspace=0.04
    )
    
    return fig, ax
    

def plot_waveform(
    waveform,
    ax: mpl.axes.Axes,
    color: str = "blue",
    alpha: float = 0.6,
    downsample: int = 10,
    sample_rate: int = 16000,
):
    x = waveform[..., ::downsample]

    new_rate = sample_rate / downsample
    x_time = torch.arange(x.shape[-1]) / new_rate

    ax.plot(x_time, x, color=color, zorder=0, alpha=alpha)
    ax.set_xlim([0, x_time[-1]])

    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    # ax.set_ylabel("waveform", fontsize=14)

def onehot_to_segment(
    va,
    vad_frame: int = 50
):
    segmentation = []
    flag = 0
    for i, vad in enumerate(va):
        if vad == 1:
            if flag == 1:
                segmentation[-1][1] += 1
            else:
                segmentation.append([i,i])
                flag = 1
        else:
            if flag == 1:
                segmentation[-1][1] += 1
                flag = 0
    result = []
    for seg in segmentation:
        start = seg[0]/vad_frame
        end = seg[1]/vad_frame
        result.append([start, end])
    return result

def plot_va(
    va,
    ax,
    color ="lightblue",
    vad_frame: int = 50,
):
    for seg in va:
        ax.axvspan(seg[0], seg[1], color=color, alpha=0.3)
            
    # waveformのフレーム数になるべき 仮処置
    # ax.set_xlim([0, 1000])
    ax.set_ylim([-1, 1])
    ax.set_yticks([])


def load_vad(vad_path):  
    with open(vad_path) as f:
        d = json.load(f)
    return d



if __name__ == "__main__":

    session = "4823"
    audio_path = os.path.join(repo_root(),"data","CallfriendENG/audio", session + ".wav")
    vad_path = os.path.join(repo_root(), "data", "CallfriendENG/vad", session + ".json")
    
    out_path = os.path.join(repo_root(), "vap_datasets", "plot", session + ".png")
    plot(audio_path, vad_path, out_path)