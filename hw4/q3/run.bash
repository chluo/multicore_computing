#!/bin/bash
#SBATCH -J myGPU           # job name
#SBATCH -o myGPU%j         # output and error file name (%j expands to jobID)
#SBATCH -n 1               # total number of mpi tasks requested
#SBATCH -p gpudev          # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 3600 seconds
#SBATCH -A EE-382C-EE361C-Multi
./cu_out 