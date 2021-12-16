# Base source author: YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Modified by ☘ Molpaxbio Co., Ltd.
# Author: jaesik won <jaesik.won@molpax.com>
# Created: 10/12/2021
# Version: 0.5
# Modified: 16/12/2021

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt
RUN pip install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache -U numpy

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app


# Usage Examples ----
# Aws configuration
#ENTRYPOINT ["aws", "configure", "set", "aws_access_key_id", "$AWS_ACCESS_KEY_ID"]
#ENTRYPOINT ["aws", "configure", "set", "aws_secret_access_key", "$AWS_SECRET_ACCESS_KEY"]

# Logging enviornment at envlog.txt
#ENTRYPOINT ["./getenv.sh", ">", "envlog.txt"]

# Training with S3 bucket and logging it at training_log.txt
#ENTRYPOINT ["./training.sh", "s3", "0"] #with gpu 0
#ENTRYPOINT ["./training.sh", "s3", "cpu"] # with cpu

# Validation with S3 bucket and logging it at val_log.txt
#ENTRYPOINT ["./infer.sh", "s3", "0"] #with gpu 0
#ENTRYPOINT ["./infer.sh", "s3", "cpu"] # with cpu

# Test with S3 bucket and logging it at test_log.txt
#ENTRYPOINT ["./test.sh", "s3", "0"] #with gpu 0
#ENTRYPOINT ["./test.sh", "s3", "cpu"] # with cpu