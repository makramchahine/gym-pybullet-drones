#!/bin/bash
N=$1 # number of runs
P=16 # number of processes

program="python culekta_follow_2d_3obj.py --plot False --gui False"

seq 1 $N | xargs -I{} -P $P sh -c "$program; echo {}"