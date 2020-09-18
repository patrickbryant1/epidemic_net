#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Visualize graph networks''')

parser.add_argument('--network1', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network2', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network3', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network4', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network5', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')

parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

###FUNCTIONS###
def visualize(network):
    '''Vizualize a network
    '''
    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Total
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))

        ax.set_xlim([0,25])
        ax.set_title('m='+str(m))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Day')
        ax.set_ylabel('Nodes left above t (%)')
        fig.tight_layout()
        fig.savefig(outdir+'deg_'+str(m)+'_total.png', format='png', dpi=300)
        plt.close()

    return None
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 5.5})
args = parser.parse_args()

network1 = np.load(args.network1[0], allow_pickle=True)
network2 = np.load(args.network1[0], allow_pickle=True)
network3 = np.load(args.network1[0], allow_pickle=True)
network4 = np.load(args.network1[0], allow_pickle=True)
network5 = np.load(args.network1[0], allow_pickle=True)

outdir = args.outdir[0]
