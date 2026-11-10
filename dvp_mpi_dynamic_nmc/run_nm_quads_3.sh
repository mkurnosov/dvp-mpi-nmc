#!/bin/bash

#
# Two servers
# 1 master process, 12 workers (one NM6408 per worker rank, server 1: /dev/NM_Quad_{00-07}, server 2: /dev/NM_Quad_{00-03})
#
mpiexec --host pine-mq1.ipa.csc.sibsutis.ru:9,pine-mq2.ipa.csc.sibsutis.ru:4 \
        --map-by core --bind-to core \
        --report-bindings \
        -x LD_LIBRARY_PATH \
        ./dvp_mpi_dynamic_nmc ../model/yolo_v8n/model_mu.nm8 ../data/video.mp4 output_mpi_q3.mp4 1



