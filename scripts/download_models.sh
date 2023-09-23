git clone https://github.com/kpu/kenlm.git
cd kenlm
apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
./compile_query_only.sh
cd ..
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
git clone https://github.com/huggingface/tokenizers
cd tokenizers/bindings/python
pip install setuptools_rust
cd ../../..
apt-get install openjdk-8-jdk
apt-get install unzip
pip install -r requirements.txt 