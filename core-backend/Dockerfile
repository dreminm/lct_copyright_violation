FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY project ./project/

RUN mkdir data

CMD python -m project.app