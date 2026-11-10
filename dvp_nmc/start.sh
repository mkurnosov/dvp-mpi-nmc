#!/bin/bash

CURDIR=`pwd`

./det ../model/yolo_v8n/model_mu.nm8 ../model/yolo_v8n/model.names ../data/video.mp4
cp -f video_out.mp4 video_out_v8n_nmc_mu.mp4







