FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /soict_hackathon

RUN apt update
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install openjdk-8-jdk -y
RUN apt-get install unzip -y
RUN apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
# ADD soict_hackathon_JointIDSF/ ./soict_hackathon_JointIDSF
ADD training/ ./training
ADD norm/ ./norm
ADD wav2vec2/ ./wav2vec2
ADD utils/ ./utils
ADD inference.py .
ADD requirements.txt .
ADD scripts/predict.sh .
ADD README.md .
ADD .dockerignore .
ADD scripts/run_commands.sh .
ADD kenlm/ ./kenlm
ADD tokenizers/ ./tokenizers

RUN pip install --upgrade pip
RUN pip install setuptools_rust

WORKDIR /soict_hackathon/kenlm
RUN sed -i -e 's/\r$//' ./compile_query_only.sh
# RUN ./compile_query_only.sh
WORKDIR /soict_hackathon

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN export PATH="$HOME/.cargo/bin:$PATH"

WORKDIR /soict_hackathon/tokenizers/bindings/python
# RUN python setup.py install
WORKDIR /soict_hackathon

RUN pip install -r requirements.txt

RUN gdown --id 1Kf_1MONyxukuYVM0rDzCHznEE0nwd4tf -O ./soict_hackathon_JointIDSF/ --folder
WORKDIR /soict_hackathon