import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=str, help="start of seed")
args = parser.parse_args()

seeds = args.seed.split(',')

l = []
for seed in seeds:
    l.append(np.load('output/output{}.npy'.format(seed)))
l = np.concatenate(l, axis=0)
print(np.average(l, axis=0))
print(np.sum(l[:,1]<l[:,3]))