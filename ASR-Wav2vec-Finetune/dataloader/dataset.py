import sys

sys.path.append("../")
import torch

from utils.feature import load_wav
from typing import Dict
import random

random.seed(42)

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
)

aug_transform_threshold = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.5),
            TimeStretch(
                min_rate=1, max_rate=1.2, leave_length_unchanged=False, p=0.5
            ),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ]
    )

class DefaultCollate:
    def __init__(self, processor, sr, noise_transform, aug_transform) -> None:
        self.processor = processor
        self.sr = sr
        self.aug_transform = aug_transform
        self.noise_transform = noise_transform
        self.duration_threshold = 8.0

    def __call__(self, inputs) -> Dict[str, torch.tensor]:
        features, transcripts = zip(*inputs)
        features, transcripts = list(features), list(transcripts)
        for i in range(len(features)):
            prob = random.random()
            if prob > 0.5:
                feature = features[i]
                features[i] = self.noise_transform(features[i], sample_rate=self.sr)
                if (feature == features[i]).all():
                    # prob = random.random()
                    # duration = len(features[i]) / self.sr            
                    # if prob > 0.5:
                    #     if duration <= self.duration_threshold:
                    #         features[i] = self.aug_transform(features[i], sample_rate=self.sr)
                    #     else:
                    #         features[i] = aug_transform_threshold(features[i], sample_rate=self.sr)
                    duration = len(features[i]) / self.sr
                    if duration <= self.duration_threshold:
                        features[i] = self.aug_transform(features[i], sample_rate=self.sr)
                    else:
                        features[i] = aug_transform_threshold(features[i], sample_rate=self.sr)


        batch = self.processor(
            features, sampling_rate=16000, padding="longest", return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor(
                transcripts, padding="longest", return_tensors="pt"
            )

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        return batch


class Dataset:
    def __init__(self, data, sr, preload_data, transform=None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item["path"], sr=self.sr)
        else:
            feature = item["wav"]

        return feature, item["transcript"]
