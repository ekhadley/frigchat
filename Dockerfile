FROM python:3.12-slim

WORKDIR /
RUN pip install bitsandbytes
RUN apt-get update && apt-get install -y build-essential
RUN pip install 'transformers[torch]'
COPY ./mistral_7b_kissy /mistral_7b_kissy
RUN pip install runpod
ENV CC=gcc
COPY ./endpoint.py /
CMD ["python3", "-u", "endpoint.py"]