from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import argparse
import uuid
import json

data_dict = {}
data_dict["audio"] = []
data_dict["transcription"] = []
data_dict["id"] = []
data_dict["entity_type"] = []
    

def map_fn(example):
    global data_dict
    example["entities_align"] = example["entities_align"].replace("\'", "\"")
    align_data = json.loads(example["entities_align"])
    # print(align_data)
    audio = example["audio"]
    for key in align_data:
        transcription = align_data[key]['d']
        words = transcription.split()
        #print(words)
        if len(words) == 1:
            continue
        x0 = align_data[key]['x0']
        x1 = align_data[key]['x1']
        
        seg = audio["array"][x0:x1]
        if (seg == []):
            print(x0)
        
        audio_dict = {"array": seg, 'path': None, "sampling_rate": audio["sampling_rate"]}
        id = uuid.uuid4()
        uuid_str = str(id)
        data_dict["audio"].append(audio_dict)
        data_dict["transcription"].append(transcription)
        data_dict["id"].append(uuid_str)
        data_dict["entity_type"].append(key)
        
    return example
        


def main(data_links, output_path, token, num_workers):
    # Load data
    ds = load_dataset(data_links)
    ds.map(map_fn, num_proc=num_workers)
    ds_entity = Dataset.from_dict(data_dict)
    print(ds_entity)
    ds_result = DatasetDict({"train": ds_entity})
    ds_result.push_to_hub(output_path, token=token)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_links', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--token', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    main(args.data_links, args.output_path, args.token, args.num_workers)