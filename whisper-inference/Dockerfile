FROM nvcr.io/nvidia/pytorch:23.12-py3

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONPATH /opt/application/
ENV PATH /opt/application/:$PATH
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/application/

COPY requirements.txt /opt/application/

RUN pip install -r requirements.txt

RUN useradd -g users user

COPY --chown=user:users project /opt/application/project

CMD python -m project.inference_kserve