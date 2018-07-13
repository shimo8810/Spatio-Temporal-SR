"""
新規実験 作成 スクリプト
"""
import os
from os import path
import argparse

# dir関連
FILE_PATH = path.dirname(path.abspath(__file__))
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

def main():
    """
    entry point
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, help="name of experiment.")
    args = parser.parse_args()

    os.mkdir(path.join(ROOT_PATH, 'experiments', args.name))
    os.mkdir(path.join(ROOT_PATH, 'results', args.name))

    print("created:", path.join(ROOT_PATH, 'experiments', args.name))
    print("created:", path.join(ROOT_PATH, 'results', args.name))

if __name__ == "__main__":
    main()
