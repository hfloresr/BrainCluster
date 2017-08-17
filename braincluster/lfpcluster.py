#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn import preprocessing
from scipy.io import loadmat
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster


class LFPCluster(object):
    """docstring for LFPCluster"""

    def __init__(self, data, rate, bad_channels=None):
        super(LFPCluster, self).__init__()
        self.Z = data
        self.nchannels = Z.shape[1]
        self.channels = {ch for ch in range(self.nchannels)}
        self.bad_channels = bad_channels
        self.rate = rate

    def standardize_lfp(self, nepochs):
        # Remove bad channels
        if self.bad_channels:
            self.Z = np.delete(self.Z, self.bad_channels, axis=1)
        for i in range(nepochs):
            ioffset = i * rate
            preprocessing.scale(self.Z[ioffset:ioffset+rate, ], copy=False)
        self.Z_norm = self.Z

    def standardize_post_lfp(self, nepochs):
        # Exclude channels for certain epochs.
        # NOTE: Highly specific to this project.
        # TODO: Have .yml or .json file to determine epoch-channel excludes
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

        good_channels = self.channels.difference(set(self.bad_channels))
        for i in range(nepochs):
            chns = good_channels
            if i in epochs_exc_chs_11_15_16:
                chns.difference({10, 14, 15})
            elif i in epochs_exc_chs_15_16:
                chns.difference({14, 15})
            elif i in epochs_exc_chs_16:
                chns.difference({15})
            elif i in epochs_exc_chs_11:
                chns.difference({10})
            chns = list(chns)  # for fancy indexing
            ioffset = i * self.rate
            self.Z[ioffset:ioffset+self.rate, chns] = preprocessing.scale(
                self.Z[ioffset:ioffset+rate, chns])


    def get_clusters(self, k, epoch, criter='maxclust'):
        my_chs = self.channels.difference(set(self.bad_channels))
        my_chs = list(my_chs)

        i = epoch - 1  # indices for epochs are 0-based
        ioffset = i * self.rate
        Z_clust = linkage(self.Z[ioffset:ioffset+rate, my_chs].T,
                          'complete', 'correlation')
        clusters = fcluster(Z_clust, k, criterion=criter)

        self.my_clusters = list(clusters)
        for ch in self.bad_channels:
            self.my_clusters.insert(ch, 0)  # Assign bad channels to cluster 0
        return self.my_clusters
