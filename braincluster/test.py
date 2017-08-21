#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from lfpcluster import LFPCluster
from scipy.io import loadmat

import matplotlib.pyplot as plt
import seaborn as sns
CMAP = plt.cm.hsv_r



# Load dataset
data = loadmat('../data/F141020-lfp-5min-1kHz.mat')

Z_pre = data['pre_pmcao']    # Extract pre-stroke data
Z_post = data['post_pmcao']  # Extract post-stroke data

# Bad channels
bad_channels = np.array([5, 8, 9, 12, 16, 26])

rate = 1000
pre_cluster = LFPCluster(Z_pre, rate, bad_channels)

num_epochs = 300
pre_cluster.standardize_lfp(num_epochs)
my_clusters = pre_cluster.get_clusters(k=4, epoch=1)
print(my_clusters)

#pre_cluster.plot_clusters(epoch=1)

epochs_exc_chs_11_15_16 = {53, 62, 63, 113, 114, 115, 116, 135, 136,
                           137, 138, 139, 140, 150, 151, 152, 153, 160,
                           161, 162, 163, 164, 165, 166, 167, 181, 182,
                           183, 184, 185, 186, 199, 200, 201, 202, 203,
                           204, 205, 206, 207, 222, 223, 236, 237, 238,
                           239, 295, 296}
epochs_exc_chs_15_16 = {61, 64, 65}
epochs_exc_chs_11_16 = {235, 240, 242}
epochs_exc_chs_16 = {170, 171}
epochs_exc_chs_11 = {118}	
						    
post_clust = LFPCluster(Z_post, rate, bad_channels)

for i in range(52, 55):
    if i in epochs_exc_chs_11_15_16:
        ex_chs = {10, 14, 15}
    elif i in epochs_exc_chs_15_16:
        ex_chs = {14, 15}
    elif i in epochs_exc_chs_11_16:
        ex_chs = {10, 15}
    elif i in epochs_exc_chs_16:
        ex_chs = {15}
    elif i in epochs_exc_chs_11:
        ex_chs = {10}
    else:
        ex_chs = None

    my_post_clusters = post_clust.get_clusters(k=4, epoch=i, ex_chs=ex_chs)
    post_clust.plot_clusters(epoch=i, clusters=my_post_clusters)
    fname = 'post_cluster_epoch_{}.png'.format(i+1)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


