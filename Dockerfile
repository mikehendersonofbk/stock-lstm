FROM python:3.7

RUN mkdir -p /opt

COPY . /opt/

WORKDIR /opt
ENTRYPOINT tail -f /dev/null