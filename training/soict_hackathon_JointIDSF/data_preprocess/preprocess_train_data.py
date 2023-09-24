from datasets import load_dataset, Dataset, DatasetDict
import random
random.seed(42)
import py_vncorenlp
py_vncorenlp.download_model(save_dir='./')
from args import args
import re
import json 
import os
from collections import defaultdict

def normalize_example(example):
    for i in ['sentence', 'sentence_annotation']:
        example[i] = re.sub(r'\s+', ' ', example[i])  # Replace multiple spaces with a single space
        example[i] = example[i].strip()  # Remove leading and trailing spaces
        example[i] = example[i].replace('.', '')  # Remove dot
        example[i] = example[i].replace(',', '')  # Remove comma
    return example

def word_segmentation(example):
    example['sentence'] = rdrsegmenter.word_segment(example['sentence'])[0]
    entities = example['entities']
    for entity in entities:
        entity['filler'] = ' '.join(rdrsegmenter.word_segment(entity['filler']))
    return example

def vietnamese_intent_to_slug(intent: str) -> str:
    intent_mapping = {
        'bật thiết bị': 'turn_on_device',
        'giảm mức độ của thiết bị': 'decrease_device_level',
        'giảm nhiệt độ của thiết bị': 'decrease_device_temperature',
        'giảm âm lượng của thiết bị': 'decrease_device_volume',
        'giảm độ sáng của thiết bị': 'decrease_device_brightness',
        'hủy hoạt cảnh': 'cancel_scene',
        'kiểm tra tình trạng thiết bị': 'check_condition_device',
        'kích hoạt cảnh': 'activate_scene',
        'mở thiết bị': 'open_device',
        'tăng mức độ của thiết bị': 'increase_device_level',
        'tăng nhiệt độ của thiết bị': 'increase_device_temperature',
        'tăng âm lượng của thiết bị': 'increase_device_volume',
        'tăng độ sáng của thiết bị': 'increase_device_brightness',
        'tắt thiết bị': 'turn_off_device',
        'đóng thiết bị': 'close_device'
    }

    return intent_mapping.get(intent, 'unknown_intent')

def intent_mapping(example):
    example['intent'] = vietnamese_intent_to_slug(example['intent'])
    return example

def generate_bio_labels(example):
    # Split the text into tokens
    tokens = example['sentence'].split()
    # Initialize labels with 'O' for each token
    labels = ['O'] * len(tokens)

    # For each entity, generate the corresponding BIO label
    for entity in example['entities']:
        entity_type = entity['type']
        entity_tokens = entity['filler'].split()

        # Find the starting index of the entity in the text
        start_idx = None
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                start_idx = i
                break

        if start_idx is not None:
            labels[start_idx] = 'B-' + entity_type.replace(" ", "_")
            for j in range(1, len(entity_tokens)):
                labels[start_idx + j] = 'I-' + entity_type.replace(" ", "_")

    return {'labels': ' '.join(labels)}


def write_column_to_txt(dataset, column_name, output_file):
    """
    Write the contents of a specific column from a Huggingface dataset to a txt file.

    Args:
    - dataset (Dataset): The loaded Huggingface dataset.
    - column_name (str): The name of the column whose contents need to be written to the txt file.
    - output_file_name (str): The path and name of the output txt file.
    """

    with open(output_file, 'w', encoding='utf-8') as file:
        for sample in dataset[column_name]:
            file.write(str(sample) + '\n')

if __name__ == "__main__":
    # load data
    data = load_dataset("json", data_files=args.data_raw_path)['train']

    # train test split 
    grouped_by_intent = defaultdict(list)
    for sample in data:
        grouped_by_intent[sample['intent']].append(sample)

    train_samples = []
    val_samples = []

    val_ratio = 0.08
    for intent, samples in grouped_by_intent.items():
        random.shuffle(samples)

        num_val_samples = int(len(samples) * val_ratio)

        val_samples.extend(samples[:num_val_samples])
        train_samples.extend(samples[num_val_samples:])

    train_dataset = Dataset.from_dict({k: [dic[k] for dic in train_samples] for k in train_samples[0]})
    val_dataset = Dataset.from_dict({k: [dic[k] for dic in val_samples] for k in val_samples[0]})

    # Create a dataset dict
    data = DatasetDict({
        'train': train_dataset,
        'test': val_dataset
    })

    # normalize text
    data_normalized = data.map(normalize_example)
    # word segermentation
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./')
    data_segmented = data_normalized.map(word_segmentation)
    # intent mapping
    data_mapped = data_segmented.map(intent_mapping)
    # generate BIO labels
    data_labeled = data_mapped.map(generate_bio_labels)
    
    final_data = data_labeled.remove_columns(["sentence_annotation"])
    # save to file
    final_data_train = final_data['train']
    final_data_dev = final_data['test']
    
    # prepare 3 text files for training
    if not os.path.exists(args.data_processed_path):
        os.mkdir(args.data_processed_path)
        os.chdir(args.data_processed_path)
        os.mkdir("word-level")
        os.chdir("word-level")
        os.mkdir("train")
        os.mkdir("dev")
        
    write_column_to_txt(final_data_train, column_name="sentence", output_file= "./train/seq.in")
    write_column_to_txt(final_data_train, column_name="labels", output_file="./train/seq.out")
    write_column_to_txt(final_data_train, column_name="intent", output_file="./train/label")

    write_column_to_txt(final_data_dev, column_name="sentence", output_file="./dev/seq.in")
    write_column_to_txt(final_data_dev, column_name="labels", output_file="./dev/seq.out")
    write_column_to_txt(final_data_dev, column_name="intent", output_file="./dev/label")

    # intent_label.txt
    intent_list = list(set(final_data['train'][:]['intent']))
    intent_list.append('UNK')

    # Write to a .txt file
    with open("intent_label.txt", "w") as file:
        for intent in sorted(intent_list):
            file.write(str(intent) + "\n")

    # slot_label.txt
    def get_unique_labels(data):
        unique_labels = set()
        for entry in data:
            labels = entry['labels'].split()
            unique_labels.update(labels)
        return list(unique_labels)

    slot_list = get_unique_labels(final_data['train'])
    slot_list.append('UNK')
    slot_list.append('PAD')

    # Write to a .txt file
    with open("slot_label.txt", "w") as file:
        for slot in slot_list:
            file.write(str(slot) + "\n")