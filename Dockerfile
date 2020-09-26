FROM woongjae94/union:base

MAINTAINER <WoongJae> <skydnd0304@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive

COPY . ./home/union
RUN apt-get update
RUN apt-get install -y libjpeg8-dev imagemagick libv4l-dev
RUN apt-get update && apt-get install git cmake make build-essential \
    libjpeg-dev imagemagick subversion libv4l-dev checkinstall libjpeg8-dev libv4l-0 -y
RUN git clone https://github.com/jacksonliam/mjpg-streamer.git
COPY . .
RUN make USE_LIBV4L2=true clean all
RUN make install

RUN cat docker-start.sh
RUN chmod +x docker-start.sh









