import pandas as pd
import numpy as np
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib.pyplot as plt

from torch_geometric.data import Data


def Graph_to_Grid(mask_path, graph, prop_ind, prop, out_path, name):
    """
    Convert Graph data to Petrel friendly data (*.GRDECL)

    Args:
    mask_path - template to restore initial dimensions for Petrel.
           Graph can be smaller due to -1 values in grids which are not necessary for graphs. (hint for Gleb: 1919 -> 1920)
    graph - graph which should be converted to *.GRDECL
    prop_ind - Graph.x[prop_ind] - index of a propert in the graph
    prop - name of the property (poro, perm, etc)
    out_path - path to save the file

    Out:
    *.GRDECL file
    """
    mask = pd.read_csv(mask_path)
    mask = pd.DataFrame(mask.iloc[:, 0]).to_numpy().reshape(1, -1)
    index_list = np.argwhere(mask[0] == -1).reshape(1, -1)
    a = graph.x[:, prop_ind].cpu().detach().numpy()
    for i in index_list[0]:
        a = np.insert(a, i, -1)
    #print(a.shape)

    # header part
    header = 'FILEUNIT                               -- Generated : VAE \n' \
             'METRIC / \n' \
             + prop + '                                   -- Generated : VAE \n' \
                      '-- Property name in VAE:' + prop
    # end of header

    np.savetxt(out_path + prop + name + '.GRDECL', a, delimiter=' ', header=header, footer='/', comments='')
    print('you can find your file here:', out_path)


def load_grid(path):
    with open(path) as f:
        lines = f.readlines()

    lines = list(map(str.strip, lines))
    lines = list(filter(None, lines))

    starts_with_dashes = [n for n, line in enumerate(lines) if line.startswith('--')]
    last_dash_line = starts_with_dashes[-1]

    data_lines = lines[last_dash_line + 1:]

    data = []
    for line in data_lines:
        numbers = line.split()
        for number in numbers:
            if number == '/':
                break

            if '*' in number:
                repeats, value = number.split('*')
                data.extend([float(value)] * int(repeats))

                continue

            data.append(float(number))

    side = int(len(data) ** 0.5)
    # data = np.array(data).reshape(side, side)

    return data


def show_grid(x, label, edge_index, path_prefix, filename):
    graph = Data(x=x, label=label, edge_index=edge_index)

    Graph_to_Grid(path_prefix + 'Facies.csv', graph, 0, 'poro', path_prefix + 'Generated_grids\\', filename)
    a = load_grid(path_prefix + 'Generated_grids\\' + 'poro' + filename + '.GRDECL')
    z = np.ones((16, 12, 10))
    for i in range(16):
        for j in range(12):
            for k in range(10):
                z[i][j][k] = a[i * 16 + j * 12 + k]
    plt.matshow(np.mean(z, axis=2))
    plt.show()

def plot_cumulative_maps(property_data, cp, ci, prod_data, NUM, path_prefix):
    """
    print to file map of well locations and cumulative oil production

    Args:
    property_data - map of property (poro, perm, whatever)
           
    cp - coords of production wells
    ci - coodrs of imj wells
    prod_data - production data
    NUM - # of iteration during interpolation (need for file name)
    path_prefix - path of the img file

    """
    #well location map
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.imshow(property_data[:,:,0]) #plot first layer
    k = 0
    for i in ci:
        k+=1
        ax1.text(i[1]-1.5, i[0]-1.5,f'I{k}', c='r')
    k = 0
    for i in cp:
        k+=1
        ax1.text(i[1]-1.5, i[0]-1.5,f'P{k}', c='r')
    
    #bubble map
    #ax2.subplot(1, 2, 2)
    ax2.imshow(property_data[:,:,0]) #plot first layer
    k = 0
    for i in ci:
        #ax2.scatter(i[1]-1.5, i[0]-1.5, s=k*20, c='red', alpha=1, edgecolors="black", linewidth=1)
        k+=1

    k = 0
    for i in cp:
        ax2.scatter(i[1]-1.5, i[0]-1.5, s=prod_data[k]*2, c='red', alpha=1, edgecolors="black", linewidth=1) #(m3/day)*4. 4 - just for bigger size
        k+=1
    #fig = plt.figure(figsize=(10, 10))
    plt.savefig(path_prefix+'grid_imgs/'+str(NUM), dpi=100)
    print('you can find your img file here:',  path_prefix+'grid_imgs/')