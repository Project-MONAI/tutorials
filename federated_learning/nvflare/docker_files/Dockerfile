FROM projectmonai/monai:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -qq update
RUN apt-get install -qq -y zip

RUN python -m pip install --upgrade pip
RUN python -m pip install nvflare==1.0.0
