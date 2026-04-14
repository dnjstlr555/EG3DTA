FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U openmim && \
    mim install mmengine "mmcv>=2.0.1"

RUN pip install joblib==1.4.2 scikit-learn==1.3.2 openpyxl==3.1.5

WORKDIR /workspace

RUN git clone https://github.com/dnjstlr555/EG3DTA.git

WORKDIR /workspace/EG3DTA

CMD ["python", "docker-test.py"]