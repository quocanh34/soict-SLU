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
from denoiser_fb.denoiser_fb import Denoiser
from text_correction.text_correction import TextCorrector

# Add ASR transcription
def add_asr_transcription(example):
    try:
        wav2vec2_finetuned.model.to(wav2vec2_finetuned.device)
        input_values = wav2vec2_finetuned.processor.feature_extractor(
            example["audio"]["array"],
            sampling_rate=example["audio"]["sampling_rate"],
            return_tensors="pt"
        ).to(wav2vec2_finetuned.device)

        with torch.no_grad():
            raw_denoised_logits = denoise.refactor_audio(example["audio"])
            denoised_audio_array = denoise.denoise_audio(raw_denoised_logits)
            
            input_values_denoise = wav2vec2_finetuned.processor.feature_extractor(
                denoised_audio_array[0],
                sampling_rate=example["audio"]["sampling_rate"],
                return_tensors="pt"
            ).to(wav2vec2_finetuned.device)

            logits = wav2vec2_finetuned.model(**input_values).logits
            logits_with_denoise = wav2vec2_finetuned.model(**input_values_denoise).logits

        prediction = wav2vec2_finetuned.processor.decode(logits.cpu().detach().numpy()[0], beam_width=500).text
        prediction_with_denoise = wav2vec2_finetuned.processor.decode(logits_with_denoise.cpu().detach().numpy()[0], beam_width=100).text

        print(f"prediction: {prediction} | with length {len(prediction)}")
        print(f"prediction_with_denoise: {prediction_with_denoise} | with length {len(prediction_with_denoise)}")

        if len(prediction_with_denoise) <= len(prediction):
            final_prediction = prediction
        else:
            final_prediction = prediction_with_denoise

        corrected_final_prediction = text_corrector.correct_text(final_prediction)

        example["pred_str"] = corrected_final_prediction
        
        

        del input_values
        del logits
        del logits_with_denoise
        torch.cuda.empty_cache()

    except Exception as e:
      print(e)
      example["pred_str"] = "N/A"

    # # Empty cuda
    # del input_values
    # del logits
    # del logits_with_denoise
    # torch.cuda.empty_cache()
    return example

# bias_list = "giờ\nphút\n%\ngarage | gara | ga ra | ca ra\ncompact | com pác | com pắc\ncafe | cà phê\nwc | vê kép xê\ngym | jim | dim | rim"

# Add norm of ASR transcription
def add_norm(example):
    bias_list = "giờ\nphút\n%\ngarage | gara | ga ra | ca ra\ncompact | com pác | com pắc\ncafe | cà phê\nwc | vê kép xê\ngym | jim | dim | rim"
    example['pred_str_norm'] = format_text(example['pred_str'].lower(), bias_list)
    print(example['pred_str_norm'])
    return example

if __name__ == '__main__':

    # Load ASR model and processor
    wav2vec2_finetuned = Wav2Vec2_finetuned(model_path=args.model_path, revision=args.revision)
    wav2vec2_finetuned.get_processor()
    wav2vec2_finetuned.get_model()
    wav2vec2_finetuned.get_device()

    # Load Denoise model
    denoise = Denoiser()
    denoise.get_device()
    denoise.get_model()

    # Load Text Correction model
    text_corrector = TextCorrector()
    text_corrector.get_device()
    text_corrector.get_model(args.text_correction_path)
    
    # Load dataset
    data = load_dataset(args.dataset_path, use_auth_token=args.token)

    # Map transcription and norm
    result = data[args.split].map(add_asr_transcription, num_proc=int(args.num_proc))
    result = result.map(add_norm, num_proc=int(args.num_proc))

    #Remove unneeded columns
    result.remove_columns(['audio'])
    result_dict = DatasetDict({"train": result})

    # print(result_dict)
    result_dict.save_to_disk(args.local_infer_result_path)

    # Push the result
    # result_dict.push_to_hub(args.hgf_infer_result_path, token=args.token) 