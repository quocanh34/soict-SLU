import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform
import librosa
import numpy as np
import random

class AudioTransform(BasicTransform):
    """Transform for Audio task"""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

class MelSpectrogram(AudioTransform):
    """Shifting time axis"""
    def __init__(self, parameters, always_apply=False, p=0.5):
        super(MelSpectrogram, self).__init__(always_apply, p)

        self.parameters = parameters
    
    def apply(self, data, **params):
        sound = data
        sr = 16000
        melspec = librosa.feature.melspectrogram(sound, sr=sr, **self.parameters)
        melspec = librosa.power_to_db(melspec)
        melspec = melspec.astype(np.float32)

        return melspec, sr

class SpecAugment(AudioTransform):
    """Shifting time axis"""
    def __init__(self, num_mask=2, freq_masking=0.15, time_masking=0.20, always_apply=False, p=0.5):
        super(SpecAugment, self).__init__(always_apply, p)

        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking
    
    def apply(self, data, **params):
        melspec, sr = data

        spec_aug = self.spec_augment(melspec, 
                                     self.num_mask,
                                     self.freq_masking,
                                     self.time_masking,
                                     melspec.min())
        


        return spec_aug, sr
    
    # Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
    def spec_augment(self, 
                    spec: np.ndarray,
                    num_mask=2,
                    freq_masking=0.15,
                    time_masking=0.20,
                    value=0):
        spec = spec.copy()
        num_mask = random.randint(1, num_mask)
        for i in range(num_mask):
            all_freqs_num, all_frames_num  = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0:f0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[:, t0:t0 + num_frames_to_mask] = value

        return spec