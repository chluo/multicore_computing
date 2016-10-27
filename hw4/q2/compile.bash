#!/bin/bash

nvcc -arch=compute_35 -code=sm_35 -o cu_out q2.cu