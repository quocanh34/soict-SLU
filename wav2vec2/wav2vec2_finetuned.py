import datasets
import torch
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers.file_utils import cached_path, hf_bucket_url

class Wav2Vec2_finetuned():
    def __init__(self, model_path, revision):
        self.model_path = model_path
        self.revision = revision

    def get_processor(self):
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("nguyenvulebinh/wav2vec2-large-vi-vlsp2020")
        self.processor = processor

    def get_model(self):
        model = SourceFileLoader("model", cached_path(hf_bucket_url(self.model_path,filename="model_handling.py"))).load_module().Wav2Vec2ForCTC.from_pretrained(self.model_path, revision=self.revision)
        self.model = model

    def get_device(self):
        current_device = torch.cuda.current_device()
        if torch.cuda.is_available():
            self.device = f"cuda:{current_device}"
        else:
            self.device = "cpu"
