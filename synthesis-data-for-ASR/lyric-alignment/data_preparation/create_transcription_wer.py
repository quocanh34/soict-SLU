from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM
import torchaudio
import torch
import argparse
from datasets import load_dataset, load_metric
import re
from jiwer import wer

# Load model & processor
model_name = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
model = SourceFileLoader("model", cached_path(hf_bucket_url(model_name,filename="model_handling.py"))).load_module().Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda:0")
processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
wer_metric = load_metric("wer")
chars_to_ignore = r'[,?.!\-;:"“\']'

def map_fn(batch):    
    audio = torch.tensor(batch["audio"]["array"])
    # print(audio)
    input_values = processor.feature_extractor(
        audio,
        sampling_rate=batch["audio"]["sampling_rate"],
        return_tensors="pt"
    ).to("cuda:0")

    with torch.no_grad():
        logits = model(**input_values).logits


    transcription = processor.tokenizer.decode(logits.argmax(dim=-1)[0].detach().cpu().numpy())
    batch["w2v2_large_transcription"] = transcription
    batch["sentence_norm_v2"] = re.sub(chars_to_ignore, '', batch["sentence_norm_v2"]).lower()
    # batch["wer"] = wer_metric.compute(predictions=batch["w2v2_large_transcription"], references=batch["sentence_norm_v2"])
    batch["wer"] = wer(batch["sentence_norm_v2"], batch["w2v2_large_transcription"])*100
    # print(batch["wer"])
    return batch

def main(data_links, output_path, token, num_workers):
    ds = load_dataset(data_links)
    ds_transcript = ds.map(map_fn, num_proc=num_workers)
    print(ds_transcript["train"][0])
    ds_transcript.push_to_hub(output_path, token=token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_links', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--token', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    main(args.data_links, args.output_path, args.token, args.num_workers)

