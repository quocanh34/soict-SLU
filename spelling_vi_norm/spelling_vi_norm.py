from vinorm import TTSnorm
from spelling_oov import download_tokenizer_files, oov_spelling
from transformers import EncoderDecoderModel
from importlib.machinery import SourceFileLoader
from regtag import reoov
import datasets
from datasets import load_dataset

## Load model & tokenizer
cache_dir='./cache'
model_name='nguyenvulebinh/spelling-oov'

path_list = download_tokenizer_files(cache_dir, model_name)

spell_tokenizer = SourceFileLoader("envibert.tokenizer",path_list[0]).load_module().RobertaTokenizer(path_list[0].replace("/envibert_tokenizer.py", ""))
spell_model = EncoderDecoderModel.from_pretrained(model_name)

error_sample_list = []
def norm(text):
    return TTSnorm(text)

def check_contain_number(text):
    characters_to_remove = ['/', '|', "'\'", '-', '.', ',', '(', ')', '[', ']' '+', ' ']
    
    for char in characters_to_remove:
        text = text.replace(char, '')
    return text

def process_spelling(text): 
    words = text.split(" ")
    processed_words = []
    for word in words:
        if reoov.check_oov_word(word) and (check_contain_number(word).isdigit() == False):
            processed_words.append(oov_spelling(word, spell_tokenizer, spell_model)[0])
        else:
            processed_words.append(word)
    new_text = " ".join(processed_words)
    new_text = norm(new_text)
    return new_text

def add_column_sentence_norm(example):
    example["sentence_norm"] = process_spelling(example["sentence"].lower())
    return example

# def filter_non_value(dataset):
#     return len(dataset['origin_transcription_norm']) != 0

# train_data = load_dataset("quocanh34/soict_train_data_w2v2_norm_WER", use_auth_token="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")

# result = train_data['train'].map(add_column_sentence_norm, num_proc=1)

# result = result.remove_columns(["w2v2_transcription", "w2v2_WER", "norm_transcription", "norm_WER"])

# result = result.train_test_split(test_size=0.1)

# result.push_to_hub("quocanh34/asr_spoken_norm_train_data", token="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")

print(process_spelling("hỗ trợ tôi tăng cái đèn cây ở ngoài hành lang lên 72% với"))