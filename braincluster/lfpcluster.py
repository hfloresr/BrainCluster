#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt
import seaborn as sns


class LFPCluster(object):
    """docstring for LFPCluster"""

    def __init__(self, Z, rate, bad_channels=None):
        super(LFPCluster, self).__init__()
        self.Z = Z
        self.nchannels = self.Z.shape[1]
        self.channels = {ch for ch in range(self.nchannels)}
        self.bad_channels = bad_channels
        self.good_channels = self._get_good_channels()
        self.rate = rate

    def _get_good_channels(self):
        if self.bad_channels is None:
            return self.channels
        return self.channels.difference(set(self.bad_channels))


    def standardize_lfp(self, nepochs):
        # Retrieve good/valid channels
        chns = list(self.good_channels)

        for i in range(nepochs):
            ioffset = i * self.rate
            self.Z[ioffset:ioffset+self.rate, chns] = preprocessing.scale(
                self.Z[ioffset:ioffset+self.rate, chns])

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

        for i in range(nepochs):
            chns = self.good_channels
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
                self.Z[ioffset:ioffset+self.rate, chns])


    def get_clusters(self, k, epoch, criter='maxclust'):
        # Retrieve good/valid channels
        chns= list(self.good_channels)

        i = epoch - 1  # indices for epochs are 0-based
        ioffset = i * self.rate
        Z_clust = linkage(self.Z[ioffset:ioffset+self.rate, chns].T,
                          'complete', 'correlation')
        clusters = fcluster(Z_clust, k, criterion=criter)

        self.my_clusters = list(clusters)
        for ch in self.bad_channels:
            self.my_clusters.insert(ch, 0)  # Assign bad channels to cluster 0
        return self.my_clusters

    def _init_grid(self):
        y = [1, 2, 3, 4] * 8
        x = [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4 + [6]*4 + [7]*4 + [8]*4
        return x, y

    def plot_clusters(self, epoch, clusters=None, cmap=plt.cm.hsv_r):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        clusts = self.my_clusters if clusters is None else clusters
        seq = [c*100 if c == 0 else 100 for c in clusts]  # clustering sequence
        x, y = self._init_grid()
        ax.scatter(x, y, c=clusts, s=seq, cmap=cmap)

        ch_labels = [ch for ch in range(1, self.nchannels+1)]
        for i, txt in enumerate(ch_labels):
            ax.annotate(txt, (x[i], y[i]))

        title = 'Epoch {}'.format(epoch)
        title_font = {'fontname': 'DejaVu Sans', 'size': '16',
                      'color': 'black', 'weight': 'bold'}
        ax.set_title(title, **title_font)
        ax.invert_yaxis()
        plt.show()
