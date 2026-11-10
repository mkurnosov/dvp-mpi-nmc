#!/bin/bash

#
# One server
# 1 master process, 4 workers (one NM6408 per worker rank, /dev/NM_Quad_{00-03})
#
mpiexec --host pine-mq1.ipa.csc.sibsutis.ru:5 \
        --map-by core --bind-to core \
        --report-bindings \
        -x LD_LIBRARY_PATH \
        ./dvp_mpi_static_nmc ../model/yolo_v8n/model_mu.nm8 ../data/video.mp4 output_mpi_q1.mp4 1





