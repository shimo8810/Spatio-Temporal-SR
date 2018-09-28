"""
データセットを生成するスクリプト
"""
import os
from os import path
import argparse
import numpy as np


#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# プロジェクトのルートパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))
# データセットディレクトリのパス
DS_PATH = path.join(ROOT_PATH, 'datasets')
#

def main():
    pass

if __name__ == '__main__':
    main()
