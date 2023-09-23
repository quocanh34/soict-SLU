import datasets
import torch
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from datasets import load_dataset, DatasetDict
from norm.infer import format_text
from jiwer import wer
from utils.args import args
from wav2vec2.wav2vec2_finetuned import Wav2Vec2_finetuned

# Add ASR transcription
def add_asr_transcription(example):
    wav2vec2_finetuned.model.to(wav2vec2_finetuned.device)
    input_values = wav2vec2_finetuned.processor.feature_extractor(
        example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        return_tensors="pt"
    ).to(wav2vec2_finetuned.device)

    with torch.no_grad():
        logits = wav2vec2_finetuned.model(**input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    example["pred_str"] = wav2vec2_finetuned.processor.decode(logits.cpu().detach().numpy()[0], beam_width=100).text

    # Empty cuda
    del input_values
    del logits
    torch.cuda.empty_cache()
    return example

# Add norm of ASR transcription
def add_norm(example):
    example['pred_str_norm'] = format_text(example['pred_str'].lower(), "giờ, phút, %")
    return example


if __name__ == '__main__':

    # Load model and processor
    wav2vec2_finetuned = Wav2Vec2_finetuned(model_path=args.model_path)
    wav2vec2_finetuned.get_processor()
    wav2vec2_finetuned.get_model()
    wav2vec2_finetuned.get_device()

    # Load dataset
    data = load_dataset(args.dataset_path, use_auth_token=args.token)
    
    # Map transcription and norm
    result = data[args.split].map(add_asr_transcription, num_proc=int(args.num_proc))
    result = result.map(add_norm, num_proc=int(args.num_proc))

    #Remove unneeded columns
    result.remove_columns(['audio'])
    result_dict = DatasetDict({"train": result})

    print(result_dict)
    result_dict.save_to_disk(args.local_infer_result_path)

    # Push the result
    # result_dict.push_to_hub(args.hgf_infer_result_path, token=args.token) 