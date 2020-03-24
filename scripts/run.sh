#!/usr/bin/env bash

docker run -it --name dprp_con --rm -p 8888:8888 -v $(pwd):/project/ dprp_image
