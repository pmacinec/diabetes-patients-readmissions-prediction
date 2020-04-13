#!/usr/bin/env bash

if [ "$1" = "-b" ]
then
  docker build -t dprp_image .
fi

docker run -it --name dprp_con --rm -p 8888:8888 -v $(pwd):/project/ dprp_image
