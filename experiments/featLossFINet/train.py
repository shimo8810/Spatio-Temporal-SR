"""
train
"""
import os
import sys
from os import path
import argparse
import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import nets as N

# パス関連
FILE_PATH = path.dirname(path.abspath(__file__))
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../../'))

sys.path.append(path.join(ROOT_PATH, 'libs'))
import datasets

def main():
    a  = datasets.Datasets()

if __name__ == "__main__":
    main()
