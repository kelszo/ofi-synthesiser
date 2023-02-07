FROM nvcr.io/nvidia/pytorch:23.01-py3

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

RUN apt-get update && apt-get -y update

RUN apt-get install -y \
    graphviz \
    swig \
    libgl1 \
    tini  \
    wget 

RUN pip install --no-cache-dir \
    catboost \
    lightgbm \
    optuna \
    sdv 

COPY jupyter_notebook_config.json .
EXPOSE 6080
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--config=./jupyter_notebook_config.json"]