# :zap: FINETUNE WAV2VEC 2.0 FOR SPEECH RECOGNITION
This code is fork from [ASR-Wav2vec-Finetune
](https://github.com/khanld/ASR-Wav2vec-Finetune) from github [khanld](https://github.com/khanld) for the baseline training pipeline, and then be updated and optimized base on the requirements of the competiton.
The core update including:
- Custom to add train wav2vec2 large model
- Custom data loader for augmentation on fly with specific 0.5 probability
- Add data augmentation strategies from audimentations library
- Script for data and noise data getting

### Table of contents
1. [Installation](#installation)
2. [Train](#train)

<a name = "installation" ></a>
### Installation
```
pip install -r requirements.txt
```

<a name = "train" ></a>
### Train
1. Prepare your dataset
    - Your dataset can be in <b>.txt</b> or <b>.csv</b> format.
    - <b>path</b> and <b>transcript</b> columns are compulsory. The <b>path</b> column contains the paths to your stored audio files, depending on your dataset location, it can be either absolute paths or relative paths. The <b>transcript</b> column contains the corresponding transcripts to the audio paths. 
    - Check out our [example files](examples/train_data_examples/) for more information.
    * <b>Important:</b> Ignoring these following notes is still OK but can hurt the performance.
        - <strong>Make sure that your transcript contains words only</strong>. Numbers should be converted into words and special characters such as ```r'[,?.!\-;:"“%\'�]'``` are removed by default,  but you can change them in the [base_dataset.py](base/base_dataset.py) if your transcript is not clean enough. 
        - If your transcript contains special tokens like ```bos_token, eos_token, unk_token (eg: <unk>, [unk],...) or pad_token (eg: <pad>, [pad],...))```. Please specify it in the [config.toml](config.toml) otherwise the Tokenizer can't recognize them.
    - Running command to prepare the dataset:
    ```
    cd utils
    python get_dataset
    ```
    The default command above will download the data from [thanhduycao/data_soict_train_synthesis_entity](https://huggingface.co/datasets/thanhduycao/data_soict_train_synthesis_entity), which is the final train data after data synthesis of our team, the method of synthesis is written in the report, and provide code in the repo.
    
    If you want to use custom dataset, this is the command:
    ```
    cd utils
    python get_dataset --dataset_name <your_hf_dataset> --id_name <id_col> --sentence_name <transcription_col>
    ```
2. Configure the [config.toml](config.toml) file: You should add hf token, hf authentication as well as hf repo in the huggingface part to push the model to hub. You can also modify batch size, eval step in here.
3. Run
- Sh command
    ```
    chmod +x asr_train.sh
    ./asr_train.sh
    ```
Detail script if run sh fail:
- Create save folder
    ```
    mkdir wav2vec2_finetune_aug_on_fly
    ```
- Run command
    ```
    cd ASR-Wav2vec-Finetune

    python train.py -c config.toml --noise_path "noise_data/RIRS_NOISES/pointsource_noises/" --pretrained_path "nguyenvulebinh/wav2vec2-large-vi-vlsp2020" --scheduler_type "cosine" --spec_aug False
    ```