FROM projectmonai/monai@sha256:766425ff41102f8b3aa98ffe0516447208dea72166dd52b4eea5239ac5469146

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -qq update
RUN apt-get install -qq -y zip

RUN python -m pip install --upgrade pip
RUN python -m pip install nvflare==1.0.0
