FROM python:3.7-slim

RUN apt update \
  && apt -y upgrade \
  && apt install -y --no-install-recommends build-essential git wget libopencv-dev \
  && apt clean \
  && apt autoclean \
  && apt autoremove \
  && rm -rf /tmp/* /var/tmp/* \
  && rm -rf /var/lib/apt/lists/* \
  rm -rf /var/lib/apt/lists/*

# install ffmpeg, ffprobe
WORKDIR /tmp
RUN wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.1.4-amd64-static.tar.xz \
  && tar xvf ffmpeg-4.1.4-amd64-static.tar.xz \
  && cp ffmpeg-4.1.4-amd64-static/ffmpeg /usr/local/bin/ \
  && cp ffmpeg-4.1.4-amd64-static/ffprobe /usr/local/bin/ \
  && pip install -U pip \
    torch==1.2.0 \
    scikit-video \
    scikit-learn \
    matplotlib \
    numpy \
    opencv-python

# create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
USER user

# all users can use /home/user as their home directory
RUN chmod 777 /home/user
WORKDIR /home/user/work
