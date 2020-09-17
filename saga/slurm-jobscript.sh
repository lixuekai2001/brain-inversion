#!/bin/bash
# properties = {properties}
module purge
module load iimpi/2019b
export MUMPS_OOC_TMPDIR=/tmp/
export SINGULARITYENV_DISPLAY=:99.0
export SINGULARITYENV_PYVISTA_OFF_SCREEN=true
export SINGULARITYENV_YVISTA_USE_PANEL=false
{exec_job}
