import sys

sys.path.append("../")
import torch

from utils.feature import load_wav
from typing import Dict
import random
import librosa

random.seed(42)

from audiomentations import (
    Compose,
    OneOf,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    RoomSimulator,
)

from augspec import SpecAugment, MelSpectrogram

aug_transform_threshold = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.5),
            TimeStretch(
                min_rate=1, max_rate=1.2, leave_length_unchanged=False, p=0.5
            ),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ]
    )

melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 20,
        "fmax": 32000
    }

melspectrogram_inverse_parameters = {
        "fmin": 20,
        "fmax": 32000
    }

transform = MelSpectrogram(parameters=melspectrogram_parameters, p=1.0)
transform_aug = SpecAugment(freq_masking=0.05, time_masking=0.1, p=0.5)

class DefaultCollate:
    def __init__(self, processor, sr, light_transform, heavy_transform, spec_aug) -> None:
        self.processor = processor
        self.sr = sr
        self.light_transform = light_transform
        self.heavy_transform = heavy_transform
        self.duration_threshold = 8.0
        self.spec_aug = spec_aug

    def __call__(self, inputs) -> Dict[str, torch.tensor]:
        features, transcripts, wer = zip(*inputs)
        features, transcripts, wer = list(features), list(transcripts), list(wer)
        for i in range(len(features)):
            # prob = random.random()
            # if prob > 0.5:
            #     feature = features[i]
            #     features[i] = self.noise_transform(features[i], sample_rate=self.sr)
            #     if (feature == features[i]).all():
            #         # prob = random.random()
            #         # duration = len(features[i]) / self.sr            
            #         # if prob > 0.5:
            #         #     if duration <= self.duration_threshold:
            #         #         features[i] = self.aug_transform(features[i], sample_rate=self.sr)
            #         #     else:
            #         #         features[i] = aug_transform_threshold(features[i], sample_rate=self.sr)
            #         duration = len(features[i]) / self.sr
            #         if duration <= self.duration_threshold:
            #             features[i] = self.aug_transform(features[i], sample_rate=self.sr)
            #         else:
            #             features[i] = aug_transform_threshold(features[i], sample_rate=self.sr)
            if wer[i] > 0 and wer[i] <= 30:
                if (self.spec_aug == True):
                    prob = random.random()
                    if prob > 0.25:
                        try:
                            features[i] = self.light_transform(features[i], sample_rate=self.sr)
                        except:
                            print("light transform error")
                    else:
                        melspec, sr = transform(data=features[i])['data']
                        data = melspec, sr
                        specAug, sr = transform_aug(data=data)['data']
                        # Convert mel spectrogram to linear scale spectrogram
                        linear_spec = librosa.feature.inverse.mel_to_stft(specAug, **melspectrogram_inverse_parameters)
                        
                        n_iter = 32
                        # Invert the spectrogram to waveform using Griffin-Lim
                        audio = librosa.griffinlim(linear_spec, n_iter=n_iter)
                        features[i] = audio
                else:
                    try:
                        features[i] = self.light_transform(features[i], sample_rate=self.sr)
                    except:
                        print("light transform error")


            elif wer[i] == 0:
                try:
                    features[i] = self.heavy_transform(features[i], sample_rate=self.sr)
                except:
                    print("heavy transform error")


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

        return feature, item["transcript"], item["wer"]
