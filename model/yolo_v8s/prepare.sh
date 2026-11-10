#!/bin/sh

nmdlp_image_converter_console frame.bmp frame08 640 640 rgb 255.0 255.0 255.0 0.0 0.0 0.0
nmdlp_compiler_console ONNX model.onnx model_su.nm8 0 
nmdlp_compiler_console ONNX model.onnx model_mu.nm8 1 

