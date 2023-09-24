# 1. Inference
## * Note: Checkpoints links
- ASR: https://huggingface.co/thanhduycao/wav2vec2-large-finetune-aug-on-fly-synthesis-fix-60-epoch-v2/ (revision info in predict.sh)
- Spoken-norm: https://huggingface.co/linhtran92/finetuned_taggenv2_55epoch_encoder_embeddings
- NLU: file JointIDSF_PhoBERTencoder.zip in folder training/soict_hackathon_JointIDSF/

## 1.0 Run all using .sh file
```bash
chmod +x scripts/predict.sh
./scripts/predict.sh
```
## * Note: In case step 1.0 doesn't work, follow these commands:
## 1.1 Run ASR and spoken-norm model
```bash
python3 inference.py --dataset_path="quocanh34/soict_test_dataset" --model_path="thanhduycao/wav2vec2-finetune-aug-on-fly-60-epoch-ver-02" --norm_path="linhtran92/finetuned_taggenv2_60epoch_encoder_embeddings" --token="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" --hgf_infer_result_path="quocanh34/test_result" --local_infer_result_path="asr_norm_result_data" --num_proc=1 --split="train" --revision="fb695560bcb4edb57659f86930dddd959516b650"
```
## 1.2 Run NLU model
```bash
cd training/soict_hackathon_JointIDSF
```
```bash
unzip JointIDSF_PhoBERTencoder.zip
```
```bash
python3 ./data_preprocess/prep_data_infer.py --data_path="asr_norm_result_data" --text_column="pred_str_norm" --split_name="train"
```
```bash
python3 predict.py  --input_file="input.txt" --output_file="predictions.jsonl" --model_dir="./new_nlu_50ep_final/4e-5/0.15/100"
```
```bash
cd ../..
```

# 2. Training
## 2.1 Train ASR
More training instructions details are in README.md of this folder
```bash
cd training/ASR-Wav2vec-Finetune
chmod +x asr_train.sh
./asr_train.sh
```
## 2.2 Train Spoken-norm
More training instructions details are in README.md of this folder
```bash
cd training/norm-tuned
chmod +x norm_train.sh
./norm_train.sh
```
## 2.3 Train NLU 
More training instructions details are in README.md of this folder
```bash
cd training
chmod 755 -R soict_hackathon_JointIDSF
cd training/soict_hackathon_JointIDSF
chmod +x nlu_train.sh
./nlu_train.sh
```
## * Reproduce new checkpoint links:
- ASR: https://huggingface.co/thanhduycao/wav2vec2-finetine-large-synthesis-validate 
    - should delete argument "--revision" in predict.sh for new model

- Spoken-norm: https://huggingface.co/linhtran92/finetuned_taggenv2__encoder_embeddings

- NLU: checkpoint in folder "training/soict_hackathon_JointIDSF/jointIDSF_PhoBERTencoder" 
    - should make it a zip file for the next infer (folder to zip is jointIDSF_PhoBERTencoder/)
    - move it to folder soict_hackathon_JointIDSF 
    - in predict.sh, also need to change the "--model_dir" name after being unzip.
    - for example: "./new_nlu_50ep_final/4e-5/0.15/100" -> "./<new_model_dir>/4e-5/0.15/100"


# 3. ADDITIONAL (synthesis data)
## 3.1 Installation
```bash
cd synthesis-data-for-ASR
pip install -r requirements.txt
```
## 3.2 Create data
```bash
python lyric-alignment/predict.py --dataset_links="quocanh34/data_for_synthesis" --output_path=<new-dataset-path> --token=<> --num_workers=1
```
