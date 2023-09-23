from transformers import EncoderDecoderModel
from importlib.machinery import SourceFileLoader
from huggingface_hub import hf_hub_download
import torch
import os

source_path_list = []

def download_tokenizer_files(cache_dir, model_name):
    resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
    for item in resources:
        if not os.path.exists(os.path.join(cache_dir, item)):
            tmp_file = hf_hub_download(repo_id=model_name, filename=item, cache_dir=cache_dir)
            
            current_working_dir = os.getcwd()

            source_path = os.path.relpath(tmp_file, current_working_dir)
            source_path_list.append(source_path)

    return source_path_list        

def oov_spelling(word, spell_tokenizer, spell_model, num_candidate=1):
    result = []
    inputs = spell_tokenizer([word.lower()])
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask)
    }
    outputs = spell_model.generate(**inputs, num_return_sequences=num_candidate)
    for output in outputs.cpu().detach().numpy().tolist():
        result.append(spell_tokenizer.sp_model.DecodePieces(spell_tokenizer.decode(output, skip_special_tokens=True).split()))
    return result   

