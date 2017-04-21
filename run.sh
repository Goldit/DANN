#!/bin/bash
nvidia-docker run --rm -it --volume=$PWD:/notebooks  -p 8888:8888 -p 80:6006 tensorflow/tensorflow:1.1.0-rc1-gpu-py3 bash
