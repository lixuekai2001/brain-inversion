#!/bin/bash
# properties = {properties}
module purge
module load iimpi/2019b
#module load OpenBLAS/0.3.7-GCC-8.3.0
export MUMPS_OOC_TMPDIR=/tmp/
export SINGULARITYENV_PYVISTA_OFF_SCREEN=true
export SINGULARITYENV_YVISTA_USE_PANEL=false
#export SINGULARITYENV_OPENBLAS_NUM_THREADS=8
#export OPENBLAS_NUM_THREADS=8
#export OMP_PLACES=threads 
#export OMP_PROC_BIND=spread
{exec_job}
