python3 ./data_preprocess/aug_data.py
python3 ./data_preprocess/preprocess_train_data.py --data_raw_path augmented_data_unique.jsonl \
                                            --data_processed_path data_aug_full_0919_22

./run_jointIDSF_PhoBERTencoder.sh