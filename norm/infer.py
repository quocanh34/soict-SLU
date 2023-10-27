import torch
import norm.model_handling as model_handling
from norm.data_handling import DataCollatorForNormSeq2Seq
from norm.model_handling import EncoderDecoderSpokenNorm
import os
import random
import norm.data_handling as data_handling
from transformers import LogitsProcessorList, StoppingCriteriaList, BeamSearchScorer
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
import norm.utils
import re
import datasets
from datasets import load_dataset
from utils.args import args

use_gpu = False
if use_gpu:
    if not torch.cuda.is_available():
        use_gpu = False

tokenizer = model_handling.init_tokenizer()
model = EncoderDecoderSpokenNorm.from_pretrained(args.norm_path).eval()
data_collator = DataCollatorForNormSeq2Seq(tokenizer)

if use_gpu:
    model = model.cuda()

def make_batch_input(text_input_list):
    batch_src_ids, batch_src_lengths = [], []
    for text_input in text_input_list:
        src_ids, src_lengths = [], []
        for src in text_input.split():
            src_tokenized = tokenizer(src)
            ids = src_tokenized["input_ids"][1:-1]
            src_ids.extend(ids)
            src_lengths.append(len(ids))
        src_ids = torch.tensor([0] + src_ids + [2])
        src_lengths = torch.tensor([1] + src_lengths + [1]) + 1
        batch_src_ids.append(src_ids)
        batch_src_lengths.append(src_lengths)
        assert sum(src_lengths - 1) == len(src_ids), "{} vs {}".format(sum(src_lengths), len(src_ids))
    input_tokenized = tokenizer.pad({"input_ids": batch_src_ids}, padding=True)
    input_word_length = tokenizer.pad({"input_ids": batch_src_lengths}, padding=True)["input_ids"] - 1
    return input_tokenized['input_ids'], input_tokenized['attention_mask'], input_word_length


def make_batch_bias_list(bias_list):
    if len(bias_list) > 0:
        bias = data_collator.encode_list_string(bias_list)
        bias_input_ids = bias['input_ids']
        bias_attention_mask = bias['attention_mask']
    else:
        bias_input_ids = None
        bias_attention_mask = None

    return bias_input_ids, bias_attention_mask


def build_spoken_pronounce_mapping(bias_list):
    list_pronounce = []
    mapping = dict({})
    for item in bias_list:
        pronounces = item.split(' | ')[1:]
        pronounces = [tokenizer(item)['input_ids'][1:-1] for item in pronounces]
        list_pronounce.extend(pronounces)    
    subword_ids = list(set([item for sublist in list_pronounce for item in sublist]))
    mapping = {item: [] for item in subword_ids}
    for item in list_pronounce:
        for wid in subword_ids:
            if wid in item:
                mapping[wid].append(item)
    return mapping

def find_pivot(seq, subseq):
    n = len(seq)
    m = len(subseq)
    result = []
    for i in range(n - m + 1):
        if seq[i] == subseq[0] and seq[i:i + m] == subseq:
            result.append(i)
    return result

def revise_spoken_tagging(list_tags, list_words, pronounce_mapping):
    if len(pronounce_mapping) == 0:
        return list_tags
    result = []
    for tags_tensor, sen in zip(list_tags, list_words):
        tags = tags_tensor.detach().numpy().tolist()
        sen = sen.detach().numpy().tolist()
        candidate_pronounce = dict({})
        for idx in range(len(tags)):
            if tags[idx] != 0 and sen[idx] in pronounce_mapping:
                for pronounce in pronounce_mapping[sen[idx]]:
                    pronounce_word = str(pronounce)
                    start_find_idx = max(0, idx - len(pronounce))
                    end_find_idx = idx + len(pronounce)
                    find_idx = find_pivot(sen[start_find_idx: end_find_idx], pronounce)
                    if len(find_idx) > 0:
                        find_idx = [item + start_find_idx for item in find_idx]
                        for map_idx in find_idx:
                            if candidate_pronounce.get(map_idx, None) is None:
                                candidate_pronounce[map_idx] = len(pronounce)
                            else:
                                candidate_pronounce[map_idx] = max(candidate_pronounce[map_idx], len(pronounce))
        for idx, len_word in candidate_pronounce.items():
            tags_tensor[idx] = 1
            for i in range(1, len_word):
                tags_tensor[idx + i] = 2
        result.append(tags_tensor)
    return result


def make_spoken_feature(input_features, text_input_list, pronounce_mapping=dict({})):
    features = {
        "input_ids": input_features[0],
        "word_src_lengths": input_features[2],
        "attention_mask": input_features[1],
        # "bias_input_ids": bias_features[0],
        # "bias_attention_mask": bias_features[1],
        "bias_input_ids": None,
        "bias_attention_mask": None,
    }
    if use_gpu:
        for key in features.keys():
            if features[key] is not None:
                features[key] = features[key].cuda()
        
    encoder_output = model.get_encoder()(**features)
    spoken_tagging_output = torch.argmax(encoder_output[0].spoken_tagging_output, dim=-1)
    spoken_tagging_output = revise_spoken_tagging(spoken_tagging_output, features['input_ids'], pronounce_mapping)
    
    # print(spoken_tagging_output)
    # print(features['input_ids'])
    word_src_lengths = features['word_src_lengths']
    encoder_features = encoder_output[0][0]
    list_spoken_features = []
    list_pre_norm = []
    for tagging_sample, sample_word_length, text_input_features, sample_text in zip(spoken_tagging_output, word_src_lengths, encoder_features, text_input_list):
        spoken_feature_idx = []
        sample_words = ['<s>'] + sample_text.split() + ['</s>']
        norm_words = []
        spoken_phrase = []
        spoken_features = []
        if tagging_sample.sum() == 0:
            list_pre_norm.append(sample_words)
            continue
        for idx, word_length in enumerate(sample_word_length):
            if word_length > 0:
                start = sample_word_length[:idx].sum()
                end = start + word_length
                if tagging_sample[start: end].sum() > 0 and sample_words[idx] not in ['<s>', '</s>']:
                    # Word has start tag
                    if (tagging_sample[start: end] == 1).sum():
                        if len(spoken_phrase) > 0:
                            norm_words.append('<mask>[{}]({})'.format(len(list_spoken_features), ' '.join(spoken_phrase)))
                            spoken_phrase = []
                            list_spoken_features.append(torch.cat(spoken_features))
                            spoken_features = []
                    spoken_phrase.append(sample_words[idx]) 
                    spoken_features.append(text_input_features[start: end])
                else:
                    if len(spoken_phrase) > 0:
                        norm_words.append('<mask>[{}]({})'.format(len(list_spoken_features), ' '.join(spoken_phrase)))
                        spoken_phrase = []
                        list_spoken_features.append(torch.cat(spoken_features))
                        spoken_features = []
                    norm_words.append(sample_words[idx])
        if len(spoken_phrase) > 0:
            norm_words.append('<mask>[{}]({})'.format(len(list_spoken_features), ' '.join(spoken_phrase)))
            spoken_phrase = []
            list_spoken_features.append(torch.cat(spoken_features))
            spoken_features = []
        list_pre_norm.append(norm_words)
        
        
    list_features_mask = []
    if len(list_spoken_features) > 0:
        feature_pad = torch.zeros_like(list_spoken_features[0][:1, :])
        max_length = max([len(item) for item in list_spoken_features])
        for i in range(len(list_spoken_features)):
            spoken_length = len(list_spoken_features[i])
            remain_length = max_length - spoken_length
            device = list_spoken_features[i].device
            list_spoken_features[i] = torch.cat([list_spoken_features[i], 
                                                 feature_pad.expand(remain_length, feature_pad.size(-1))]).unsqueeze(0)
            list_features_mask.append(torch.cat([torch.ones(spoken_length, device=device, dtype=torch.int64),
                                                 torch.zeros(remain_length, device=device, dtype=torch.int64)]).unsqueeze(0))
    if len(list_spoken_features) > 0:
        list_spoken_features = torch.cat(list_spoken_features)
        list_features_mask = torch.cat(list_features_mask)
    
    return list_spoken_features, list_features_mask, list_pre_norm


def make_bias_feature(bias_raw_features):
    features = {
        "bias_input_ids": bias_raw_features[0],
        "bias_attention_mask": bias_raw_features[1]
    }
    if use_gpu:
        for key in features.keys():
            if features[key] is not None:
                features[key] = features[key].cuda()
    return model.forward_bias(**features)


def decode_plain_output(decoder_output):
    plain_output = [item.split()[1:] for item in tokenizer.batch_decode(decoder_output['sequences'], skip_special_tokens=False)]
    scores = torch.stack(list(decoder_output['scores'])).transpose(1, 0)
    logit_output = torch.gather(scores, -1, decoder_output['sequences'][:, 1:].unsqueeze(-1)).squeeze(-1)
    special_tokens = list(tokenizer.special_tokens_map.values())
    generated_output = []
    generated_scores = []
    # filter special tokens
    for out_text, out_score in zip(plain_output, logit_output):
        temp_str, tmp_score = [], []
        for piece, score in zip(out_text, out_score):
            if piece not in special_tokens:
                temp_str.append(piece)
                tmp_score.append(score)
        if len(temp_str) > 0:
            generated_output.append(' '.join(temp_str).replace('▁', '|').replace(' ', '').replace('|', ' ').strip())
            generated_scores.append((sum(tmp_score)/len(tmp_score)).cpu().detach().numpy().tolist())
        else:
            generated_output.append("")
            generated_scores.append(0)
    return generated_output, generated_scores


def generate_spoken_norm(list_spoken_features, list_features_mask, bias_features):
    @dataclass
    class EncoderOutputs(ModelOutput):
        last_hidden_state: torch.FloatTensor = None
        hidden_states: torch.FloatTensor = None
        attentions: torch.FloatTensor = None

    batch_size = list_spoken_features.size(0)
    max_length = 50
    device = list_spoken_features.device
    decoder_input_ids = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
    stopping_criteria = model._get_stopping_criteria(max_length=max_length, max_time=None,
                                                     stopping_criteria=StoppingCriteriaList())
    model_kwargs = {
        "encoder_outputs": EncoderOutputs(last_hidden_state=list_spoken_features),
        "encoder_bias_outputs": bias_features,
        "attention_mask": list_features_mask
    }
    decoder_output = model.greedy_search(
        decoder_input_ids,
        logits_processor=LogitsProcessorList(),
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
        **model_kwargs,
    )
    plain_output, plain_score = decode_plain_output(decoder_output)
    # plain_output = tokenizer.batch_decode(decoder_output['sequences'], skip_special_tokens=True)
    # # print(decoder_output)
    # plain_output = [word.replace('▁', '|').replace(' ', '').replace('|', ' ').strip() for word in plain_output]
    return plain_output, plain_score


def generate_beam_spoken_norm(list_spoken_features, list_features_mask, bias_features, num_beams=3):
    @dataclass
    class EncoderOutputs(ModelOutput):
        last_hidden_state: torch.FloatTensor = None

    batch_size = list_spoken_features.size(0)
    max_length = 50
    num_return_sequences = 1
    device = list_spoken_features.device
    decoder_input_ids = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
    stopping_criteria = model._get_stopping_criteria(max_length=max_length, max_time=None,
                                                     stopping_criteria=StoppingCriteriaList())
    model_kwargs = {
        "encoder_outputs": EncoderOutputs(last_hidden_state=list_spoken_features),
        "encoder_bias_outputs": bias_features,
        "attention_mask": list_features_mask
    }
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device,
        do_early_stopping=True,
        num_beam_hyps_to_keep=num_return_sequences,
    )
    decoder_input_ids, model_kwargs = model._expand_inputs_for_generation(
        decoder_input_ids, expand_size=num_beams, is_encoder_decoder=True, **model_kwargs
    )

    decoder_output = model.beam_search(
        decoder_input_ids,
        beam_scorer,
        logits_processor=LogitsProcessorList(),
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=None,
        return_dict_in_generate=True,
        **model_kwargs,
    )

    plain_output = tokenizer.batch_decode(decoder_output['sequences'], skip_special_tokens=True)
    plain_output = [word.replace('▁', '|').replace(' ', '').replace('|', ' ').strip() for word in plain_output]
    return plain_output, None


def reformat_normed_term(list_pre_norm, spoken_norm_output, spoken_norm_output_score=None, threshold=None, debug=False):
    output = []
    for pre_norm in list_pre_norm:
        normed_words = []
        # words = pre_norm.split()
        for w in pre_norm:
            if w.startswith('<mask>'):
                term = w[7:].split('](')
                # print(w)
                # print(term)
                term_idx = int(term[0])
                norm_val = spoken_norm_output[term_idx]
                norm_val_score = None if (spoken_norm_output_score is None or threshold is None) else spoken_norm_output_score[term_idx]
                pre_norm_val = term[1][:-1]
                if debug:
                    if norm_val_score is not None:
                        normed_words.append("({})({:.2f})[{}]".format(norm_val, norm_val_score, pre_norm_val))
                    else:
                        normed_words.append("({})[{}]".format(norm_val, pre_norm_val))
                else:
                    if threshold is not None and norm_val_score is not None:
                        if norm_val_score > threshold:
                            normed_words.append(norm_val)
                        else:
                            normed_words.append(pre_norm_val)
                    else:
                        normed_words.append(norm_val)
            else:
                normed_words.append(w)
        output.append(" ".join(normed_words))
    return output


def infer(text_input_list, bias_list):
    # extract bias feature
    bias_raw_features = make_batch_bias_list(bias_list)
    bias_features = make_bias_feature(bias_raw_features)
    pronounce_mapping = build_spoken_pronounce_mapping(bias_list)

    # Chunk split input and create feature
    text_input_chunk_list = [norm.utils.split_chunk_input(item, chunk_size=60, overlap=20) for item in text_input_list]
    num_chunks = [len(i) for i in text_input_chunk_list]
    flatten_list = [y for x in text_input_chunk_list for y in x]
    input_raw_features = make_batch_input(flatten_list)

    # Extract norm term and spoken feature
    list_spoken_features, list_features_mask, list_pre_norm = make_spoken_feature(input_raw_features, flatten_list, pronounce_mapping)

    # Merge overlap chunks
    list_pre_norm_by_input = []
    for idx, input_num in enumerate(num_chunks):
        start = sum(num_chunks[:idx])
        end = start + num_chunks[idx]
        list_pre_norm_by_input.append(list_pre_norm[start:end])
    text_input_list_pre_norm = [norm.utils.merge_chunk_pre_norm(list_chunks, overlap=20, debug=False) for list_chunks in list_pre_norm_by_input]

    if len(list_spoken_features) > 0:
        spoken_norm_output, spoken_norm_score = generate_spoken_norm(list_spoken_features, list_features_mask, bias_features)
    else:
        spoken_norm_output, spoken_norm_score = [], None

    return reformat_normed_term(text_input_list_pre_norm, spoken_norm_output, spoken_norm_score, threshold=15, debug=False)


patterns_to_replace = {
    r'(\d+) %': r'\1%',
    r'ra +đi +ô': 'radio',
    r'[jr]im': 'gym',
    r'độ +xê': 'độ c',
    r'độ +oc': 'độ c',
    r'oc': "độ c",
    r'láp +tóp': "laptop",
    r"lét": "led",
    r" răm": " trăm",
    r'(\d+)h(\d+)p': r'\1 giờ \2 phút',
    r'(\d+)h(\d+)': r'\1 giờ \2 phút',
    r"(\d+)h": r"\1 giờ "
}

def format_text(text_input, list_bias_input):
    print('{}\n{}\n\n'.format(text_input, list_bias_input))
    bias_list = list_bias_input.strip().split('\n')
    norm_result = infer([text_input], bias_list)
    return normalize_text(norm_result[0], patterns_to_replace)

def normalize_text(sentence, patterns_to_replace):
    modified_sentence = sentence
    #test
    print(sentence)
    for pattern, replacement in patterns_to_replace.items():
        modified_sentence = re.sub(pattern, replacement, modified_sentence)
    return modified_sentence