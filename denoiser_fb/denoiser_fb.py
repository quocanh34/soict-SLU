import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from datasets import load_dataset


class Denoiser():
    def __init__(self):
        pass
    def get_model(self):
        self.model = pretrained.dns64().to(self.device)

    def refactor_audio(self, audio):
        noise_audio_array = torch.FloatTensor(audio["array"]).unsqueeze(0)
        noise_audio_sr = audio["sampling_rate"]
        raw_denoised_audio = convert_audio(noise_audio_array.to(self.device), noise_audio_sr, self.model.sample_rate, self.model.chin)
        return raw_denoised_audio

    def denoise_audio(self, raw_denoised_audio):
        denoised_audio = self.model(raw_denoised_audio)[0]
        return denoised_audio

    def get_device(self):
        try:
            current_device = torch.cuda.current_device()
            if torch.cuda.is_available():
                self.device = f"cuda:{current_device}"
        except:
            self.device = "cpu"