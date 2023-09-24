cd ASR-Wav2vec-Finetune
pip install -r requirements.txt
pip install protobuf==3.20.*
install git-lfs if not have
git-lfs install
cd utils
python get_dataset.py
python get_noise_data.py
cd ..
mkdir wav2vec2_finetune_aug_on_fly
CUDA_VISIBLE_DEVICES=0 python train.py -c config.toml --noise_path "noise_data/RIRS_NOISES/pointsource_noises/" --pretrained_path "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
cd ..

