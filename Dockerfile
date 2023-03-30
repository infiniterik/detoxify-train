# Use nvidia/cuda image
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
CMD nvidia-smi

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Versions
ARG TORCH_CUDA_VERSION=cu116
ARG TORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake \
    bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libopenmpi-dev && \
    apt-get clean
RUN unattended-upgrade
RUN apt-get autoremove -y

# Install Pythyon (3.8 as default)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN pip install -U pip
RUN pip install pygit2 pgzip

# (Optional) Intall test dependencies
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install datasets accelerate evaluate coloredlogs absl-py rouge_score seqeval scipy sacrebleu nltk scikit-learn parameterized sentencepiece
RUN pip install fairscale deepspeed mpi4py
RUN pip install wandb transformers

# Install onnxruntime-training dependencies
RUN pip install onnx ninja
RUN pip install torch==${TORCH_VERSION}+${TORCH_CUDA_VERSION} torchvision==${TORCHVISION_VERSION} -f https://download.pytorch.org/whl/cu116/torch_stable.html
RUN pip install onnxruntime-training==1.14.1 -f https://download.onnxruntime.ai/onnxruntime_stable_cu116.html
RUN pip install torch-ort
RUN pip install --upgrade protobuf==3.20.2

ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
RUN python -m torch_ort.configure

RUN pip install optimum

WORKDIR /home
#COPY requirements.txt .
#RUN pip install -r requirements.txt
ENV WANDB_API_KEY=112ecceb19501df3ee5000fd55661456501701c4

COPY . .

CMD ["/bin/bash"]