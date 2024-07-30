#!/bin/bash

docker run -it --gpus all -m 200g --shm-size=8G --name "june_container" -v "/home/vision/dongjun:/workspace" -v "/data1:/data1" -v "/data2:/data2" pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel /bin/bash
