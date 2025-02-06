from datasets import load_dataset
from torch.utils.data import DataLoader
import lhotse
from lhotse import CutSet, Fbank
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
    PerturbSpeed,
    PerturbVolume,
    SpecAugment,
    make_worker_init_fn,
)
from lhotse.recipes import (
    download_librispeech,
    prepare_librispeech,
)
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper

train_ds = load_dataset("sarulab-speech/J-CHAT", split="train")
train_urls = train_ds["text"]
train_cutset = CutSet.from_webdataset(train_urls)

for cut in train_cutset:
    print(cut)
    # audio = cut.load_audio()
    # print(len(audio[0]))

# train_sampler = DynamicBucketingSampler(
#     train_cutset,  # <-- note the "_webdataset" variant being used here
#     shuffle=False,
#     max_duration=1000.0,
#     num_buckets=10,
# )
# print("1")
# train_dataset = K2SpeechRecognitionDataset(
#     cut_transforms=[
#         PerturbSpeed(factors=[0.9, 1.1], p=2 / 3),
#         PerturbVolume(scale_low=0.125, scale_high=2.0, p=0.5),
#     ],
#     input_transforms=[
#         SpecAugment(),  # default configuration is well-tuned
#     ],
#     input_strategy=OnTheFlyFeatures(Fbank()),
# )

# print("1")

# train_iter_dataset = IterableDatasetWrapper(
#     dataset=train_dataset,
#     sampler=train_sampler,
# )
# print("1")

# train_dloader = DataLoader(
#     train_iter_dataset,
#     batch_size=None,
#     # For faster dataloading, use num_workers > 1
#     num_workers=0,
#     # Note: Lhotse offers its own "worker_init_fn" that helps properly
#     #       set the random seeds in all workers (also with multi-node training)
#     #       and sets up data de-duplication for multi-node training with WebDataset.
#     worker_init_fn=make_worker_init_fn(),
# )

# print(train_dloader)
# print("1")

# .cut_into_windows(duration=5)
# print(cutset[0])
# fbank = Fbank()
# cuts = cutset[0].compute_and_store_features(
#     extractor=fbank,
#     storage=storage,
#     storage_path="data/fbank",
#     num_jobs=8,
# )
# dataset = VadDataset(cutset)
# sampler = SimpleCutSampler(cutset, max_cuts=1000)
# dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)
# batch = next(iter(train_dloader))
# print(batch)