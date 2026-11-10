#!/bin/bash

#
# One server
# 1 master process, 8 workers (one NM6408 per worker rank, /dev/NM_Quad_{00-07})
#
mpiexec --host pine-mq1.ipa.csc.sibsutis.ru:9 \
        --map-by core --bind-to core \
        --report-bindings \
        -x LD_LIBRARY_PATH \
        ./dvp_mpi_static_nmc ../model/yolo_v8n/model_mu.nm8 ../data/video.mp4 output_mpi_q2.mp4 1




