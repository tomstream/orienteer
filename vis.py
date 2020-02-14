import matplotlib as mpl
mpl.use('Agg')
mpl.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
import numpy as np


a = '/home/liuzongtao/project/RL/Jotaro/data/gnn_urgency_IL_40000_0.1.npy'
b = np.load(a)

b = np.histogram(b, bins=np.arange(10)/1000)
plt.plot(b[1][:-1],b[0])
plt.savefig('/home/liuzongtao/project/RL/Jotaro/data/figs.pdf')