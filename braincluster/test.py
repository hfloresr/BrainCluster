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
