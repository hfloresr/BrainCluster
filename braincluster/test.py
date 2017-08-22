#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from lfpcluster import LFPCluster
from scipy.io import loadmat
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
data = loadmat('../data/F141020-lfp-5min-1kHz.mat')

Z_pre = data['pre_pmcao']    # Extract pre-stroke data
Z_post = data['post_pmcao']  # Extract post-stroke data


def plot_dendrogram(Z_clust, bad_channels=None, max_d=None):
    if bad_channels is None:
        labels = [i for i in range(1, 33)]
    else:
        labels = [i for i in range(1, 33) if i not in bad_channels]

    dend = dendrogram(Z_clust, color_threshold=1, labels=labels)
    if max_d:
        plt.axhline(y=max_d, c='k')
    plt.show()
    

# Bad channels
#bad_channels = np.array([5, 8, 9, 12, 16, 26])
bad_channels = {5, 8, 9, 12, 16, 26}

rate = 1000
pre_cluster = LFPCluster(Z_pre, rate, bad_channels)

num_epochs = 300
pre_cluster.standardize_lfp(num_epochs)
Zpre_clust, my_clusters = pre_cluster.get_clusters(k=4, epoch=1)
plot_dendrogram(Zpre_clust, bad_channels, max_d=0.8)

#pre_cluster.plot_clusters(epoch=1)

epochs_exc_chs_11_15_16 = {54, 63, 64, 114, 115, 116, 117, 136, 137,
                           138, 139, 140, 141, 151, 152, 153, 154, 161,
                           162, 163, 164, 165, 166, 167, 168, 182, 183,
                           184, 185, 186, 187, 200, 201, 202, 203, 204,
                           205, 206, 207, 208, 223, 224, 237, 238, 239,
                           240, 296, 297}
epochs_exc_chs_15_16 = {62, 65, 66}
epochs_exc_chs_11_16 = {236, 241, 243}
epochs_exc_chs_16 = {171, 172}
epochs_exc_chs_11 = {119}

post_clust = LFPCluster(Z_post, rate, bad_channels)

#for i in range(1, 301):
#    if i in epochs_exc_chs_11_15_16:
#        ex_chs = {10, 14, 15}
#    elif i in epochs_exc_chs_15_16:
#        ex_chs = {14, 15}
#    elif i in epochs_exc_chs_11_16:
#        ex_chs = {10, 15}
#    elif i in epochs_exc_chs_16:
#        ex_chs = {15}
#    elif i in epochs_exc_chs_11:
#        ex_chs = {10}
#    else:
#        ex_chs = None
#
#    my_post_clusters = post_clust.get_clusters(k=4, epoch=i, ex_chs=ex_chs)
#    post_clust.plot_clusters(epoch=i, clusters=my_post_clusters)
#    fname = 'post_cluster_epoch_{}.png'.format(i+1)
#    plt.savefig(fname, bbox_inches='tight')
#    plt.close()


