import os
from transformers import AutoTokenizer, AutoFeatureExtractor
from model_handling import Wav2Vec2ForCTC
import torch
import datasets
import utils
import sys
from tqdm import tqdm
import torchaudio
import json
import time
import csv
import argparse
import torch.multiprocessing as mp
import re
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM
import torchaudio
import torch
from datasets import load_from_disk, DatasetDict

def load_model():
    global model
    global tokenizer
    global feature_extractor
    global vocab

    model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
    if use_gpu:
        model = model.to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer.get_vocab()))]


def do_asr(waveform):
    input_values = feature_extractor.pad([
        {"input_values": feature_extractor(item, sampling_rate=16000)["input_values"][0]} for item in
        waveform
    ], return_tensors='pt')["input_values"]
    if use_gpu:
        input_values = input_values.to("cuda:0")

    out_values = model(input_values=input_values)
    logits = out_values.logits[0]

    emissions = torch.log_softmax(logits, dim=-1)
    emission = emissions.cpu().detach()
    emission[emission < -20] = -20

    emission[:,tokenizer.convert_tokens_to_ids('|')] = torch.tensor([max(item) for item in list(zip(emission[:,tokenizer.convert_tokens_to_ids('|')], emission[:,tokenizer.convert_tokens_to_ids('<pad>')]))])
    emission[:,tokenizer.convert_tokens_to_ids('<pad>')] = -20

    return emission

def do_force_align(waveform, emission, word_list, sample_rate=16000, base_stamp=0):
    i = 0
    transcript = '|'.join(word_list)
    dictionary = {c: i for i, c in enumerate(vocab)}
    tokens = [dictionary.get(c, 0) for c in transcript]
    trellis = utils.get_trellis(emission, tokens, blank_id=tokenizer.convert_tokens_to_ids('|'))
    path = utils.backtrack(trellis, emission, tokens)
    segments = utils.merge_repeats(path, transcript)
    word_segments = utils.merge_words(segments)
    word_segments = utils.add_pad(word_segments, emission)
    ratio = waveform.size(1) / (trellis.size(0) - 1)
    result = []
    score = []
    for word in word_segments:
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        segment = waveform[:, x0:x1]
        #print(len(segment[0]))

        result.append({
            "d": word.label,
            # "s": int(x0/sample_rate*1000)+base_stamp,
            # "e": int(x1/sample_rate*1000)+base_stamp,
            # "score": word.score,
            "x0": x0,
            "x1": x1,
            "idx": i,
        })

        score.append({
            "d": word.label,
            "score": word.score
        })

        i+=1

    assert [item["d"] for item in result] == word_list
    # print(result)
    return result, score

def audio_align(example):
    wav = torch.tensor(example['audio']['array']).float()
    wav = wav.view(1, -1)

    chars_to_ignore = r'[,?.!\-;:"%“\'�]'
    chars_special = {'bẩy': 'bảy'}
    example['sentence_norm_v2'] = re.sub(chars_to_ignore, '', example["sentence_norm_v2"]).lower()
    for key in chars_special:
        example['sentence_norm_v2'] = re.sub(key, chars_special[key], example["sentence_norm_v2"])
    words = example['sentence_norm_v2'].split()
    # print(example['sentence_norm'])
    # print("Norm: ", utils.norm_word(example['sentence']))
    single_words_list = [word.split() for word in words]
    single_words = [y for x in single_words_list for y in x]

    # wav to text_prob
    emission = do_asr(wav)
    # text_prob to (single_words & timestamp)
    word_piece, score = do_force_align(wav, emission, single_words)
    entities_json_file = generate_entities_json(word_piece, example['entities_norm'])
    example['entities_align'] = str(entities_json_file)
    example['entities_score'] = str(score)
    # try:
    #     entities_json_file = generate_entities_json(word_piece, example['entities_norm'])
    #     example['entities_align'] = str(entities_json_file)
    #     example['entities_score'] = str(score)
    # except:
    #     print(example['sentence'])
    #     print(example["sentence_norm"])
    #     example['entities_align'] = ""
    #     example['entities_score'] = ""
    # audio_start = word_piece[0]['x0']
    # audio_end = word_piece[-1]['x1']
    # example['audio']['array'] = example['audio']['array'][audio_start:audio_end + padding]
    
    torch.cuda.empty_cache()
    return example

def generate_entities_json(alignment_output, entities):
    output_json = {}
    alignment_dict = {output["d"]: output for output in alignment_output}
    alignment_output_cp = alignment_output.copy()
    for entity in entities:
        filler = entity["filler"]
        if " " in filler:
            # Handle multi-word fillers
            filler_words = filler.split(" ")
            filler_text = " ".join(filler_words)

            start_idx = 0
            for i in range(len(alignment_output_cp)):
                if alignment_output_cp[i]["d"] == filler_words[0]:
                    if alignment_output_cp[i+1]["d"] == filler_words[1]:
                        start_idx = i
                        break
            
            if (start_idx == 0):
                print("shit")
                print(alignment_output_cp[start_idx]["d"] + alignment_output_cp[start_idx+1]["d"])
            end_idx = start_idx + len(filler_words) - 1
            # print("Start_index: ", start_idx)
            # print("End_index: ", end_idx)
            # aligmnent_word = [output["d"] for output in alignment_output_cp]
            # print(aligmnent_word)
            start = alignment_output_cp[start_idx]["x0"]
            end = alignment_output_cp[end_idx]["x1"]

            del alignment_output_cp[start_idx : (end_idx + 1)]
                
            # print(filler_words)
            output_json[entity["type"]] = {"d": filler_text, "x0": start, "x1": end, "idx": start_idx}
        elif filler in alignment_dict:
            # Handle single-word fillers
            output_json[entity["type"]] = {"d": alignment_dict[filler]["d"], "x0": alignment_dict[filler]["x0"], "x1": alignment_dict[filler]['x1'], "idx": alignment_dict[filler]["idx"]}

    del alignment_output_cp
    return output_json

def filter_wer(example):
    return example["wer"] <= 25

def filter_non_entities_norm(example):
    return example['entities_align'] != ""

def norm_entities(example):
    entities = example["entities"]
    for entity in entities:
        entity["filler"] = utils.norm_word(entity["filler"])
    example["entities_norm"] = entities
    # print(example["entities_norm"])
    return example

def main(data_links, output_path, token, num_workers):
    # Load data
    data = datasets.load_dataset(data_links)
    data_preprocessed_path = "data_preprocessed"
    output_data_path = "data"

    # Load model
    load_model()
    
    print(data)
    if not os.path.exists(data_preprocessed_path):
        os.makedirs(data_preprocessed_path)
        data = data['train'].filter(filter_wer)
        data = data.map(norm_entities, num_proc=num_workers)
        data.save_to_disk(data_preprocessed_path)
    else:
        data = load_from_disk(data_preprocessed_path)
        
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
        aligned_data = data.map(audio_align, num_proc=num_workers)
        aligned_data = aligned_data.filter(filter_non_entities_norm)
        aligned_data.save_to_disk(output_data_path)
    else:
        aligned_data = load_from_disk(output_data_path)

    aligned_data = DatasetDict({"train": aligned_data})
    aligned_data.push_to_hub(output_path, token=token)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:24'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_links', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--token', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    use_gpu = True
    if use_gpu:
        if not torch.cuda.is_available():
            use_gpu = False

    model_path = 'nguyenvulebinh/lyric-alignment'
    model = None
    tokenizer = None
    feature_extractor = None
    vocab = None

    main(args.data_links, args.output_path, args.token, args.num_workers)