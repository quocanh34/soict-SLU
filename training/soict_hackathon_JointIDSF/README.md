# soict_hackathon_JointIDSF

## Installation 
    chmod 755 -R /soict_hackathon_JointIDSF
    cd soict_hackathon_JointIDSF/
    pip3 install -r requirements.txt

## Training and Evaluation

### Data Augmentation
    python3 ./data_preprocess/aug_data.py

### Preprocess for training  
    python3 ./data_preprocess/preprocess_train_data.py --data_raw_path augmented_data_unique.jsonl \
                                                --data_processed_path data_aug_full_0919_22

    ./run_jointIDSF_PhoBERTencoder.sh
    
## Inference
    python3 ./data_preprocess/prep_data_infer.py --data_path=<path_to_test_data> \'
                               --text_column=<text column name> \
                               --split_name=<split of data file>  

    python3 predict.py  --input_file <path_to_input_file> \
                        --output_file <output_file_name> \
                        --model_dir ./jointIDSF_PhoBERTencoder/4e-5/0.15/100