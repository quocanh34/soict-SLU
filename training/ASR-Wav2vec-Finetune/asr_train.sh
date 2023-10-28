#pip install -r requirements.txt
apt-get install git-lfs
git-lfs install
cd utils
python3 get_dataset.py
python3 get_noise_dataset.py
cd ..
mkdir wav2vec2_finetune_aug_on_fly
CUDA_VISIBLE_DEVICES=0 python3 train.py -c config.toml --noise_path "noise_data/RIRS_NOISES/pointsource_noises/" --pretrained_path "nguyenvulebinh/wav2vec2-large-vi-vlsp2020" --scheduler_type "cosine" --spec_aug False
cd ..