FROM nvcr.io/nvidia/tensorflow:21.07-tf2-py3

ADD . /workspace/rhnet
WORKDIR /workspace/rhnet

RUN pip3 install git+https://github.com/NVIDIA/dllogger
RUN pip3 install -r requirements.txt
