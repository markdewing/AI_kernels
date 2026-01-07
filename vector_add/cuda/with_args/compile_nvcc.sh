#!/bin/sh

nvcc -arch native timed_vector_add.cu arg_parser.cpp
