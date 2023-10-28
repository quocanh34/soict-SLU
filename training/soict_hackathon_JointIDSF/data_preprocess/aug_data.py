import json
import random
random.seed(42)
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
from pyvi.ViTokenizer import tokenize
import re
from collections import Counter
import copy

# Define Augmentation Process
@lru_cache(maxsize=None)
def find_similar_entity_with_simcse(entity_type, entity_value, intent):
    possible_entities = intent_entities[intent].get(entity_type, [])
    possible_entities = [e for e in possible_entities if e != entity_value]

    if not possible_entities:
        return entity_value

    source_tokenizer = tokenize(entity_value)
    embed_source = model.encode(source_tokenizer)

    similarities = [
        util.pytorch_cos_sim(embed_source, entity_to_embedding[possible_entity]).item()
        for possible_entity in possible_entities
    ]

    return possible_entities[similarities.index(max(similarities))]

def regenerate_entities_from_annotation(annotation):
    """
    Extract entities and their types from the sentence_annotation.
    Return a list of entities with their type and filler.
    """
    entity_pattern = re.compile(r'\[ ([^\]]+) : ([^\]]+) \]')
    entities = []
    for match in entity_pattern.findall(annotation):
        entity_type, filler = match
        entities.append({
            'type': entity_type,
            'filler': filler
        })
    return entities

def augment_and_correct_all(sample, augmentation_ratio=0.5):
    """
    Augment based on the sentence_annotation, reconstruct the sentence,
    and regenerate the entities list.
    """
    if random.random() > augmentation_ratio:
        return None

    augmented_sample = copy.deepcopy(sample)
    original_annotation = sample['sentence_annotation']
    annotation = original_annotation

    num_entities_to_replace = random.randint(1, len(sample['entities']))
    for entity in reversed(sample['entities'][:num_entities_to_replace]):
        similar_entity = find_similar_entity_with_simcse(entity['type'], entity['filler'], sample['intent'])
        if similar_entity != entity['filler']:
            annotation = annotation.replace(f"[ {entity['type']} : {entity['filler']} ]", f"[ {entity['type']} : {similar_entity} ]")

    # Reconstruct sentence
    sentence = re.sub(r'\[ [^\]]+ : ([^\]]+) \]', r'\1', annotation)

    if sentence == sample['sentence']:
        return None

    augmented_sample['sentence'] = sentence
    augmented_sample['sentence_annotation'] = annotation
    augmented_sample['entities'] = regenerate_entities_from_annotation(annotation)

    return augmented_sample


def augment_entry_owner(entry, chosen_word):
    device_filler = next(entity['filler'] for entity in entry['entities'] if entity['type'] == 'device')
    augmented_sentence = entry['sentence'].replace(device_filler, device_filler + " của " + chosen_word, 1)
    augmented_annotation = entry['sentence_annotation'].replace(device_filler, device_filler + " của " + chosen_word, 1)

    # Update the device entity filler
    for entity in entry['entities']:
        if entity['type'] == 'device':
            entity['filler'] = entity['filler'] + " của " + chosen_word
            break

    return {
        'sentence': augmented_sentence,
        'intent': entry['intent'],
        'sentence_annotation': augmented_annotation,
        'entities': entry['entities'],
    }

def augment_entry_num_loc(entry, chosen_word):
    device_filler = next(entity['filler'] for entity in entry['entities'] if entity['type'] == 'device')

    # Augment sentence and annotation using our approach
    augmented_sentence = entry['sentence'].replace(device_filler, device_filler + " " + chosen_word, 1)
    augmented_annotation = entry['sentence_annotation'].replace(device_filler, device_filler + " " + chosen_word, 1)

    # Update the device entity filler
    for entity in entry['entities']:
        if entity['type'] == 'device':
            entity['filler'] = entity['filler'] + " " + chosen_word
            break

    return {
        'sentence': augmented_sentence,
        'intent': entry['intent'],
        'sentence_annotation': augmented_annotation,
        'entities': entry['entities'],
    }

with open("raw_data/intent_entities.json", 'r') as f:
    intent_entities = json.load(f)

with open("raw_data/train_final_20230919.jsonl", 'r') as f:
    training_samples = [json.loads(line) for line in f.readlines()]

for sample in training_samples:
    del sample['id']
    del sample['file']

model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

# Augment by swapping entities
all_entities = set()
for intent, entities in intent_entities.items():
    for entity_type, entity_values in entities.items():
        all_entities.update(entity_values)

all_entities_tokenized = [tokenize(entity) for entity in all_entities]
entity_to_embedding = {
    entity: model.encode(tokenized_entity)
    for entity, tokenized_entity in zip(all_entities, all_entities_tokenized)
}
# 1. Analyze Intents Distribution
intent_counts = Counter([sample['intent'] for sample in training_samples])

# 2. Identify Less Frequent Intents
num_intents_to_augment = 4
less_frequent_intents = sorted(intent_counts, key=intent_counts.get)[:num_intents_to_augment]

augmented_samples = []

# 3. Augment Samples for Each Less Frequent Intent
for intent in less_frequent_intents:
    samples_for_intent = [sample for sample in training_samples if sample['intent'] == intent]

    for sample in samples_for_intent:
        augmented_sample = augment_and_correct_all(sample)
        if augmented_sample:  # Ensure the augmented sample is not None
            augmented_samples.append(augmented_sample)

seen_sentences = set()
swapped_samples = []
for sample in augmented_samples:
    sentence = sample['sentence']
    if sentence not in seen_sentences:
        seen_sentences.add(sentence)
        swapped_samples.append(sample)

# Augment by adding owner

words_after_cua_in_locations = []

for entry in training_samples:
    location_entities = [entity for entity in entry['entities'] if entity['type'] == 'location']

    for loc in location_entities:
        if "của" in loc['filler']:
            # Split the location filler at "của" and then split by spaces
            parts = loc['filler'].split("của", 1)
            if len(parts) > 1:
                words = parts[1].split()

                # Capture all words until the end or until a "]"
                phrase_after_cua = []
                for word in words:
                    if "]" in word:
                        phrase_after_cua.append(word.replace("]", ""))
                        break
                    phrase_after_cua.append(word)

                if phrase_after_cua:
                    words_after_cua_in_locations.append(" ".join(phrase_after_cua))

# Get the unique phrases after "của" in location entities
unique_phrases_after_cua = list(set(words_after_cua_in_locations))

device_entries = [entry for entry in training_samples if any(entity['type'] == 'device' for entity in entry['entities'])]
augment_count = len(device_entries) // 15
to_augment = random.sample(device_entries, augment_count)

augmented_entries_owner = [augment_entry_owner(entry, random.choice(unique_phrases_after_cua)) for entry in to_augment]

# Augment by adding number and location
# Lists to store the extracted phrases
so_numbers = []
directions = ["bên trái", "bên phải"]

# Iterate over each entry and extract the desired phrases
for entry in training_samples:
    for entity in entry.get('entities', []):
        if entity['type'] == 'location':
            filler = entity['filler']

            # Extract "số" followed by a number
            match = re.search(r'số (\d+)', filler)
            if match:
                so_numbers.append(match.group(0))

            # Extract directions "bên trái" and "bên phải"
            for direction in directions:
                if direction in filler:
                    so_numbers.append(direction)

device_entries = [entry for entry in training_samples if any(entity['type'] == 'device' for entity in entry['entities'])]
augment_count = len(device_entries) // 15
to_augment = random.sample(device_entries, augment_count)

# Use the combined augmentation function
augmented_entries_num = [augment_entry_num_loc(entry, random.choice(so_numbers)) for entry in to_augment]

# Combine all augmented samples
combined_augmented_samples = training_samples + swapped_samples + augmented_entries_owner + augmented_entries_num

combined_augmented_file_path = 'augmented_data.jsonl'
with open(combined_augmented_file_path, 'w', encoding='utf-8') as file:
    for entry in combined_augmented_samples:
        file.write(json.dumps(entry) + '\n')

with open(combined_augmented_file_path, 'r') as f:
    samples = [json.loads(line) for line in f.readlines()]

# Filter duplicates based on the 'sentence' field
seen_sentences = set()
unique_samples = [sample for sample in samples if sample['sentence'] not in seen_sentences and not seen_sentences.add(sample['sentence'])]

# Save the unique samples to a new JSONL file
cleaned_file_path = combined_augmented_file_path.replace(".jsonl", "_unique.jsonl")
with open(cleaned_file_path, 'w') as f:
    for sample in unique_samples:
        f.write(json.dumps(sample) + "\n")

print(f"Cleaned data saved to: {cleaned_file_path}")

