# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Global configuration."""

from Rignak_DeepLearning.data import get_dataset_roots

#----------------------------------------------------------------------------
# Paths.

result_dir = 'results'
data_dir = get_dataset_roots('stylegan')[0]
cache_dir = 'cache'
run_dir_ignore = ['__pycache__', '.ipynb_checkpoints', 'datasets', 'results', 'cache']

#----------------------------------------------------------------------------
