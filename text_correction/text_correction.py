from transformers import pipeline
import torch

class TextCorrector():
    def __init__(self):
        pass
    def get_model(self, model_path):
        self.model = pipeline("text2text-generation", model=model_path)

    def correct_text(self, text):
        corrected_text = self.model(text, max_length=128)[0]['generated_text']
        return corrected_text

    def get_device(self):
        try:
            current_device = torch.cuda.current_device()
            if torch.cuda.is_available():
                self.device = f"cuda:{current_device}"
        except:
            self.device = "cpu"