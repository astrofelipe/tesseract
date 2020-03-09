import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='LC and transit plot (useful for TTV priors!)')
parser.add_argument('Light curves', type=str, nargs='+', help='Light curve file(s)')
parser.add_argument('--transit', type=float, nargs=3, help='P and t0 values')

args = parser.parse_args()
print(args.keys())

fig, ax = plt.subplots(figsize=[15,3])

for f in args['Light curves']:
    print(f)
