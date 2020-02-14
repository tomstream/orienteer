import matplotlib as mpl
mpl.use('Agg')
mpl.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fn", type=str, help="fn")
parser.add_argument("--factor", type=int, default=10, help="start of seed")
args = parser.parse_args()



data = np.load(args.fn)

plt.plot(np.arange(len(data)) * args.factor, data)

plt.savefig(args.fn.split('.')[0]+'.pdf')