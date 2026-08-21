#!/bin/bash

xhost +
docker run -it --rm \
    --network=host \
    --gpus all \
    --ipc=host \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "`pwd`/../models:/home/`whoami`/models" \
    -v "`pwd`/../scripts:/home/`whoami`/scripts" \
    -v "`pwd`/../sam3tensorrt:/home/`whoami`/sam3tensorrt" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    --name sam3-tensorrt-trace \
    sam3-tensorrt:trace
    bash
xhost -
