import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import gen_data
import pickle

def get_cmap(N):
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


gen_data.generate_data(10000,50,'distance')

for idx in range(0,50):
    for fn in ['dqn_greedy','dqn','no_dqn']:
        depot, loc, prize = gen_data.get_data(idx)
        total, path = pickle.load(open('solution/{}.pkl'.format(fn), 'rb'))[idx]
        print(total, path)
        path = [[-1] + path + [-1]]
        loc0 = np.array(loc)
        loc = loc + [depot]
        loc = np.array(loc)
        prize = np.array(prize)

        op_size = len(loc)
        N = 1 #The num of path
        # depot = np.random.uniform(size=(2))
        # loc = np.random.uniform(size=(op_size, 2))
        # prize = np.random.uniform(size=(op_size))
        # path = [[1,5,3,4,2],[6,7,8]]
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        fig, ax = plt.subplots(figsize=(5.3, 5))
        # plt.title(str(N)+' routes',fontsize='large', fontweight='bold')


        cmap = get_cmap(N+1)
        col = []
        for i in range(N):
            col.append(cmap(i))
        print(col)
        print(path)

        for index, x in enumerate(path):
            l = len(x)
            for i in range(l-1):
                plt.annotate("", xy=(loc[x[i]][0], loc[x[i]][1]),xytext=(loc[x[i+1]][0], loc[x[i+1]][1]), arrowprops=dict(arrowstyle="simple", color=col[index], connectionstyle="arc3", mutation_scale =10))

        plt.scatter(loc0[:,0],loc0[:,1],s = prize*prize*40)
        plt.scatter([depot[0]], [depot[1]], s=200, color='black', marker='*', label="prize = {:.2f}".format(total))

        vpatch = []
        vlabel = []
        for i in range(N):
            vpatch.append(mpatches.Patch(color=col[i]))
            vlabel.append("the "+str(i)+"th route")
        # plt.legend(handles=vpatch, labels=vlabel)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('solution/{}_{}.pdf'.format(idx, fn))
        plt.close('all')