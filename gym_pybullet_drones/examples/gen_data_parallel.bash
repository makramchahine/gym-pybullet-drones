#!/bin/bash
N=299 # number of runs
P=16 # number of processes

program="python culekta.py --plot False --gui False"

seq 1 $N | xargs -I{} -P $P sh -c "$program; echo {}"