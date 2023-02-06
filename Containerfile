FROM nvcr.io/nvidia/pytorch:23.01-py3

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

COPY data data

COPY requirements.txt .

RUN pip install --no-cache-dir \
    catboost \
    lightgbm \
    optuna \
    sdv 

COPY ofisynthesiser ofisynthesiser