FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN mkdir -p /enteval
WORKDIR /enteval

COPY EntEval /enteval
COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT python main.py
