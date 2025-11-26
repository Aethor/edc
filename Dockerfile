FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD . /edc
WORKDIR /edc

RUN apt update -y
RUN apt install -y git dtach

RUN uv sync

CMD ["bash"]
