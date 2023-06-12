#!/bin/bash
N=32 # number of runs
P=16 # number of processes

program="python runna.py --plot False --gui False"

seq 1 $N | xargs -I{} -P $P sh -c "$program; echo {}"