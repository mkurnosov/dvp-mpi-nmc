#!/bin/bash

g++ det.cpp -o det -O3 -march=native `pkg-config --cflags --libs opencv4` -I . \
    -I $NMDLP_HOME/include -L $NMDLP_HOME/lib -lnmdlp -lnmdlp_compiler -lnmdlp_image_converter



