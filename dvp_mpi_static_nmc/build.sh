#!/bin/bash

mpicxx -O3 -march=native -Wall dvp_mpi_static_nmc.cpp -o dvp_mpi_static_nmc `pkg-config --cflags --libs opencv4` \
       -I $NMDLP_HOME/include -L $NMDLP_HOME/lib -lnmdlp -lnmdlp_compiler -lnmdlp_image_converter


