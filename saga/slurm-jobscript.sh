#!/bin/bash
# properties = {properties}
module purge
module load iimpi/2019b
export  MUMPS OOC TMPDIR=/tmp/
{exec_job}
