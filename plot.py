import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from networkx.drawing.nx_pydot import graphviz_layout
from nilearn import datasets, plotting
from scipy.sparse import coo_matrix

from utils import tree_construction, find_trunk


excel = pd.read_excel('Neuron_consensus_264.xlsx', sheet_name='264 ROIs', header=1,
                      names=['ROI', 'index', 'color', 'system'], usecols=[0, 31, 34, 36])

colors = ['#FFFFFF', '#00CCFF', '#FF6600', '#800080', '#FF99CC', '#FF0000', '#969696',
          '#0000FF', '#FFFF00', '#000000', '#993300', '#339966', '#00FF00', '#99CCFF']

plt.rcParams['savefig.dpi'] = 300

# set node properties
border_list = list()  # a list of colors for node borders
color_list = list()
for i in excel['index']:
    if i < 0:  # uncertain system
        border_list.append('#808080')  # color the borders of white nodes gray
        color_list.append(colors[0])
    else:
        border_list.append(colors[i])
        color_list.append(colors[i])

# construct the brain tree
brain_tree = tree_construction(overall_flag=True)
row, col = brain_tree.nonzero()
edge_list = np.stack((row, col), axis=0).T.tolist()

T = nx.Graph()
T.add_nodes_from(range(brain_tree.shape[0]))
T.add_edges_from(edge_list)

# plot the brain tree
plt.figure(figsize=(12.5, 5))
pos = graphviz_layout(T, prog='neato')
nx.draw(T, pos, node_size=30, width=2, node_color=color_list, edgecolors=border_list)
plt.savefig('brain_tree.png', bbox_inches='tight', format='png')

# identify hierarchical trunks
trunk_list = find_trunk(brain_tree.shape[0], edge_list, 100)

level = 1
size_list = [0] * brain_tree.shape[0]
for trunk in trunk_list[level-1]:
    for t in trunk:
        size_list[t] = 20  # the visible node

_row, _col = list(), list()
for trunk in trunk_list[level-1]:
    _row.extend(trunk[:-1])
    _col.extend(trunk[1:])
_data = [1] * len(_row)
coo = coo_matrix((_data, (_row, _col)), shape=brain_tree.shape)

power = datasets.fetch_coords_power_2011()
coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T

# plot the emotional area and its internal connections at the `level`th level
plotting.plot_connectome(
    (coo + coo.T).toarray(),
    coords,
    node_color=color_list,
    node_size=size_list,
    annotate=False,
    edge_kwargs={'alpha': 0.5, 'color': '#00FF00', 'linewidth': 3},
    output_file=f'level{level}.png'
)
