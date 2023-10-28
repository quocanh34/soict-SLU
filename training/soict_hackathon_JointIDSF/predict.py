import argparse
import logging
import os
import datasets
import pandas as pd
from datasets.load import load_from_disk

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer
import jsonlines

from datasets import Dataset

logger = logging.getLogger(__name__)


def get_device(pred_config):
    try:
        current_device = torch.cuda.current_device()
        if torch.cuda.is_available() and not pred_config.no_cuda:
            return f"cuda:{current_device}"
    except:
        return "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            args.model_dir, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args)
        )
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(
    lines,
    pred_config,
    args,
    tokenizer,
    pad_token_label_id,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset

intent_vi = {'turn_on_device': 'Bật thiết bị',
            'decrease_device_level': 'Giảm mức độ của thiết bị',
            'decrease_device_temperature': 'Giảm nhiệt độ của thiết bị',
            'decrease_device_volume': 'Giảm âm lượng của thiết bị',
            'decrease_device_brightness': 'Giảm độ sáng của thiết bị',
            'cancel_scene': 'Hủy hoạt cảnh',
            'check_condition_device': 'Kiểm tra tình trạng thiết bị',
            'activate_scene': 'Kích hoạt cảnh',
            'open_device': 'Mở thiết bị',
            'increase_device_level': 'Tăng mức độ của thiết bị',
            'increase_device_temperature': 'Tăng nhiệt độ của thiết bị',
            'increase_device_volume': 'Tăng âm lượng của thiết bị',
            'increase_device_brightness': 'Tăng độ sáng của thiết bị',
            'turn_off_device': 'Tắt thiết bị',
            'close_device': 'Đóng thiết bị'}

def process_entities(entities):
    entity_dict = {}
    for entity in entities:
        entity_type = entity["type"]
        entity_filler = entity["filler"]
        if entity_type in entity_dict:
            entity_dict[entity_type].append(entity_filler)
        else:
            entity_dict[entity_type] = [entity_filler]
    processed_entities = []
    for entity_type, fillers in entity_dict.items():
        if len(fillers) > 1:
            combined_fillers = " ".join(fillers)
            processed_entities.append({"type": entity_type, "filler": combined_fillers})
        else:
            for filler in fillers:
                processed_entities.append({"type": entity_type, "filler": filler})

    entities = processed_entities
    return entities

def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_label_ids": None,
                "slot_labels_ids": None,
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                if args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                

    # Write to output file
    opts = []
    file_names = []
    with open("ids.txt", "r") as f:
        files = f.read()
        file_names = files.split("\n")
    for words, slot_preds, intent_pred, file_name in zip(lines, slot_preds_list, intent_preds, file_names): # iterate through each sample
        """
        words :  1 sentence
        slot_preds : 0 0 0 entities 0 0 0...
        intent_pred : order of intent of that sentence in intent dict
        """

        entities = []
        for word, pred in zip(words, slot_preds):
            if pred != "O":
                pred = pred.split('-')[1] # remove B I at beginning
                entity = {"type": pred.replace('_',' '), "filler": word.replace('_',' ')}
                entities.append(entity)
        entities = process_entities(entities)
        opt = {
            "intent": intent_vi[intent_label_lst[intent_pred]],
            "entities": entities,
            "file": file_name
        }
        opts.append(opt)
    try:
      norm_dataset = load_from_disk("asr_norm_result_data")["train"]
      nlu_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=opts))

      norm_dataset = norm_dataset.add_column('intent', nlu_ds['intent'])
      norm_dataset = norm_dataset.add_column('entities', nlu_ds['entities'])
      norm_dataset = norm_dataset.add_column('file', nlu_ds['file'])

      norm_dataset.push_to_hub("quocanh34/best_tts3_new_nlu_denoise", token="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") 
    except:
      pass
      
    with jsonlines.open(pred_config.output_file, 'w') as writer:
        writer.write_all(opts)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="input.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="predictions.jsonl", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)