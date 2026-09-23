#!/bin/bash

docker build --build-arg user_id=$(id -u) --build-arg USER=$(whoami) --build-arg NAME=$(hostname) --rm -t sam3-tensorrt:infer -f Dockerfile.trt .
docker build --build-arg user_id=$(id -u) --build-arg USER=$(whoami) --build-arg NAME=$(hostname) --rm -t sam3-tensorrt:ros -f Dockerfile.ros .
