python3 inference.py --dataset_path="quocanh34/soict_test_dataset" --model_path="thanhduycao/wav2vec2-large-finetune-aug-on-fly-synthesis-fix-60-epoch-v2" --norm_path="linhtran92/finetuned_taggenv2_55epoch_encoder_embeddings" --token="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" --hgf_infer_result_path="quocanh34/test_result" --local_infer_result_path="training/soict_hackathon_JointIDSF/asr_norm_result_data" --num_proc=1 --split="train" --revision="fb695560bcb4edb57659f86930dddd959516b650"

cd training/soict_hackathon_JointIDSF

unzip JointIDSF_PhoBERTencoder.zip

python3 ./data_preprocess/prep_data_infer.py --data_path="asr_norm_result_data" --text_column="pred_str_norm" --split_name="train"
python3 predict.py  --input_file="input.txt" --output_file="predictions.jsonl" --model_dir="./JointIDSF_PhoBERTencoder/4e-5/0.15/100"