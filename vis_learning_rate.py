import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
import define
import pickle
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
mpl.rc('pdf', fonttype=42)
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, edgecolor='#000000')    # legend fontsize

define.init()


def f(line, savefn):
    save = []
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for _i, (fn, name) in enumerate(line):
        l = []
        for i in range(5):
            tmp = glob.glob('/home/liuzongtao/RL/Jotaro/result/performance{}{}_*.npy'.format(fn, i))[0]
            print(tmp)
            tmp = np.load(tmp)
            # print(tmp.shape)
            tmp = tmp[np.arange(15,140,5)]

            tmp = np.expand_dims(tmp, axis=0)
            l.append(tmp)
        cat = np.concatenate(l, axis=0)
        avg = np.average(cat, axis=0)
        std = np.std(cat, axis=0)/np.sqrt(5)

        save.append([avg, std, name])

        ax.errorbar(np.arange(1, len(avg)+1)*5, avg,fmt='-',marker='o',yerr=std, label=name,markersize=1,linewidth=2,elinewidth=1)
    pickle.dump(save, open('result/tmp_{}.pkl'.format(savefn),'wb'))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('pic/xx_{}.pdf'.format(savefn))
    plt.close('all')

f([['nohidden', '#layer=0'],['1layer','#layer=1'], ['2layer','#layer=2'], ['','#layer=4']], 'layer')
f([['8hidden','H=8'],['16hidden','H=16'], ['','H=32'], ['64hidden', 'H=64']], 'hidden')
f([['gat', 'without MHA in GAT'],['','with MHA in GAT']],'gat')