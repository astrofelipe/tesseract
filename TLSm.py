import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Run TLS over selected multisector LCs')
parser.add_argument('File', type=str, help='Multisector list')

args = parser.parse_args()
