#!/bin/bash
#SBATCH -J myGPU           # job name
#SBATCH -o myGPU%j         # output and error file name (%j expands to jobID)
#SBATCH -n 1               # total number of mpi tasks requested
#SBATCH -p gpudev          # queue (partition) -- normal, development, etc.
#SBATCH -t 00:00:20        # run time (hh:mm:ss) - 20 seconds
#SBATCH -A EE-382C-EE361C-Multi
#SBATCH --mail-user=chunheng.luo@utexas.edu # replace by your email
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
./cu_out 