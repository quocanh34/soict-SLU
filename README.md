# 1. Inference

## 1.1 Run ASR and spoken-norm model
```bash
python inference.py --dataset_path="quocanh34/soict_test_dataset" --model_path="thanhduycao/wav2vec2-finetune-aug-on-fly-60-epoch-ver-02" --norm_path="linhtran92/finetuned_taggenv2_60epoch_encoder_embeddings" --token="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" --hgf_infer_result_path="quocanh34/test_result" --local_infer_result_path="asr_norm_result_data" --num_proc=1 --split="train" 
```
## 1.2 Run NLU model
```bash
cd soict_hackathon_JointIDSF
```
```bash
unzip JointIDSF_PhoBERTencoder.zip
```
```bash
python3 ./data_preprocess/prep_data_infer.py --data_path="asr_norm_result_data" --text_column="pred_str_norm" --split_name="train"
```
```bash
python3 predict.py  --input_file="input.txt" --output_file="result.jsonl" --model_dir="./JointIDSF_PhoBERTencoder/4e-5/0.15/100"
```
```bash
cd ..
```

# 2. Training
## 2.1 Train ASR
## 2.2 Train Spoken-norm
## 2.3 Train NLU 