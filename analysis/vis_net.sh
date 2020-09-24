#!/usr/bin/env bash
N1=/home/pbryant/epicemic_net/results/New_York/n10000/10000_1_0_1_1_1_1_edges.npy
N2=/home/pbryant/epicemic_net/results/New_York/n10000/10000_2_0_1_1_1_1_edges.npy
N3=/home/pbryant/epicemic_net/results/New_York/n10000/10000_3_0_1_1_1_1_edges.npy
N4=/home/pbryant/epicemic_net/results/New_York/n10000/10000_4_0_1_1_1_1_edges.npy
N5=/home/pbryant/epicemic_net/results/New_York/n10000/10000_5_0_1_1_1_1_edges.npy

OUTDIR=../results/network_visualization/

./net_vis.py --outdir $OUTDIR
