from datasets import load_dataset, Dataset
import argparse
import random
from tqdm import tqdm
import pandas as pd
import json
import os

entities_type = ["command", "device", "location", "scene", "time at", "target number", "changing value", "duration"]
random.seed(42)

def create_entity_dict_index(dataset):
    entities_dict_index = {"command": [], "device": [], "location": [], "scene": [], "time at": [], "target number": [], "changing value": [], "duration": []}
    for i in tqdm(range(len(dataset))):
        entities = dataset[i]['entities']
        for entity in entities:
            entities_dict_index[entity["type"]].append(i)
    for key in entities_dict_index:
        entities_dict_index[key] = list(set(entities_dict_index[key]))
    return entities_dict_index

def create_sample_dict(new_audio_array, new_trans, old_trans):
    sample_dict = {"audio": {"array": new_audio_array, 'path': None, "sampling_rate": 16000}, "transcription": new_trans, "old_transcription": old_trans}
    return sample_dict

def create_synthesis_data(dataset, num_swap_entities, entities_dict_index):
    result = []
    for i in tqdm(range(len(dataset))):
        entites = dataset[i]["entities"]
        # print(entites)
        entity_json = dataset[i]["entities_align"].replace("\'", "\"")
        entities_swap = json.loads(entity_json)
        # entities_swap = dataset[i]["entities_align"]
        swap_entities = [entites[i] for i in sorted(random.sample(range(len(entites)), num_swap_entities))]
        # swap_entities = random.sample(entites, k = num_swap_entities)
        
        # print(swap_entities)
        swap_entities_name = [entity["type"] for entity in swap_entities]
        swap_entities_list = []
        for key in entities_swap:
            if key in swap_entities_name:
                swap_entities_list.append(entities_swap[key])
                print(entities_swap[key])
                if (len(swap_entities_list) == num_swap_entities):
                    break
        audio_array = dataset[i]["audio"]["array"]

        target_entities_list = []
        audio_target_list = []
        for name in swap_entities_name:
            target_entity_idx = random.choice(entities_dict_index[name])
            # print(target_entity_idx)
            # print(name)
            # print(dataset[target_entity_idx]["entities_align"])
            data_json_target = dataset[target_entity_idx]["entities_align"].replace("\'", "\"")
            entities_target_swap = json.loads(data_json_target)
            target_entities_list.append(entities_target_swap[name])
            audio_target_list.append(dataset[target_entity_idx]["audio"]["array"])
        
        if len(swap_entities_list) == 1:
            s = swap_entities_list
            t = target_entities_list
            synthesis_audio = audio_array[:s[0]["x0"]] + audio_target_list[0][t[0]["x0"]:t[0]["x1"]] + audio_array[s[0]["x1"]:]
            synthesis_sentence = dataset[i]["sentence_norm_v2"]
            for j in range(len(swap_entities_list)):
                synthesis_sentence = synthesis_sentence.replace(s[j]["d"], t[j]["d"])
            
        elif len(swap_entities_list) == 2:
            s = swap_entities_list
            t = target_entities_list
            entity_swap_1 = audio_target_list[0][t[0]["x0"]:t[0]["x1"]]
            entity_swap_2 = audio_target_list[1][t[1]["x0"]:t[1]["x1"]]
            audio_array_cp_0 = audio_array.copy()
            audio_array_cp_1 = audio_array.copy()
            audio_array_cp_2 = audio_array.copy()
            remain_0 = audio_array_cp_0[:s[0]["x0"]]
            # print(len(audio_array))
            # print(len(audio_array_cp_1))
            # print(s[0]["x1"])
            # print(s[1]["x0"])
            remain_1 = audio_array_cp_1[s[0]["x1"]:s[1]["x0"]]
            remain_2 = audio_array_cp_2[s[1]["x1"]:]
            synthesis_audio = remain_0 + entity_swap_1 + remain_1 + entity_swap_2 + remain_2
            synthesis_sentence = dataset[i]["sentence_norm_v2"]
            
            # s_word_len = len(s[0]["d"].split())
            # t_word_len = len(t[0]["d"].split())
            # if (s_word_len == t_word_len):
            #     sentence[s[0]["idx"]] = t[0]["d"]
            # else if (s_word_len > t_word_len)
            for j in range(len(swap_entities_list)):
                synthesis_sentence = synthesis_sentence.replace(s[j]["d"], t[j]["d"])
        result.append(create_sample_dict(synthesis_audio, synthesis_sentence, dataset[i]["sentence_norm_v2"]))
    return result


def main(data_links, output_path, token, num_workers):
    ds = load_dataset(data_links)
    # ds_test = ds["train"].select([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    dict_path = "data_dict"
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)
        entities_dict_index = create_entity_dict_index(ds["train"])
        json_object = json.dumps(entities_dict_index, indent=4)
 
        # Writing to sample.json
        with open(dict_path + "/dict_entity.json", "w") as outfile:
            outfile.write(json_object)
    else:
        json.loads(dict_path + "/dict_entity.json")
        
    num_swap_entities = 2
    # ds_synthesis = ds.map(map_fn, num_proc=num_workers, fn_kwargs={"num_swap_entities": num_swap_entities})\
    result = create_synthesis_data(ds["train"], num_swap_entities, entities_dict_index)
    final_dataset = Dataset.from_pandas(pd.DataFrame(data=result))
    final_dataset.push_to_hub(output_path, token=token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_links', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--token', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    main(args.data_links, args.output_path, args.token, args.num_workers)