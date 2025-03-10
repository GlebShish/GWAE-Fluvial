from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
#import chart_studio.plotly as py
import networkx as nx
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils.convert import to_networkx
import plotly.io as pio
import matplotlib.pyplot as plt


def show_me_graph_2d(dataset):
    '''
    2d visualization of a graph
    
    Args: graph of pytorch geometrical type
    
    '''
    G = to_networkx(dataset)  # convert to nx type
    pos = nx.layout.spring_layout(G)
    node_labels = dataset.label[list(G.nodes)].numpy()

    # Create Edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    i = 0
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        i += 1
        print(i / len(G.edges) * 100, end='\r')

    # create nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

        print(node / len(G.nodes) * 100, end='\r')

    # add color to node points
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = ('Name: ' + str(adjacencies[0]) + '<br>label: ' + str(node_labels[node]))
        node_trace['text'] += tuple([node_info])

        print(node / len(G.nodes) * 100, end='\r')

    pio.renderers.default = "browser"
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network Graph the Grid',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.show()
    return print('check it out 2d')


def show_me_graph_3d(dataset):
    '''
    3d visualization of a graph
    
    Args: graph of pytorch geometrical type
    
    '''
    G = to_networkx(dataset)
    pos = nx.layout.spring_layout(G, dim=3)
    node_labels = dataset.label[list(G.nodes)].numpy()

    # Create Edges
    i = 0
    edge_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_trace['z'] += tuple([z0, z1, None])
        i += 1
        print(i / len(G.edges) * 100, end='\r')

    # create nodes
    node_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    for node in G.nodes():
        x, y, z = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['z'] += tuple([z])

        print(node / len(G.nodes) * 100, end='\r')

        # add color to node points

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = ('Name: ' + str(adjacencies[0]) + '<br>label: ' + str(node_labels[node]))
        node_trace['text'] += tuple([node_info])

        print(node / len(G.nodes) * 100, end='\r')

    pio.renderers.default = "browser"

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network Graph the Grid',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.show()
    return print('check it out 3d')


def show_me_graph_property_2d(dataset, property_number):
    '''
    2d visualization of a graph
    
    Args: graph of pytorch geometrical type
    
    '''
    G = to_networkx(dataset)  # convert to nx type
    pos = nx.layout.spectral_layout(G, dim=2)
    pos_arr = np.asarray(list(pos.values()))
    node_labels = dataset.label[list(G.nodes)].numpy()
    node_property = dataset.x[:, property_number][list(G.nodes)].numpy()

    # Create Edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    x0 = pos_arr[np.asarray(list(G.edges()))[:, 0]][:, 0]
    y0 = pos_arr[np.asarray(list(G.edges()))[:, 0]][:, 1]

    x1 = pos_arr[np.asarray(list(G.edges()))[:, 1]][:, 0]
    y1 = pos_arr[np.asarray(list(G.edges()))[:, 1]][:, 1]

    Nones = np.full(len(x0), None)

    edge_trace['x'] = tuple(np.asarray(tuple(zip(x0, x1, Nones))).flatten())
    edge_trace['y'] = tuple(np.asarray(tuple(zip(y0, y1, Nones))).flatten())

    # create nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='bluered',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Property',
                xanchor='left',
                titleside='right'
            ),
            opacity=np.around(node_labels),
            line=dict(width=2)))

    node_trace['x'] = pos_arr[:, 0]
    node_trace['y'] = pos_arr[:, 1]

    # add color to node points
    node_trace['marker']['color'] = node_property
    for node, adjacencies in enumerate(G.adjacency()):
        node_info = ('Name: ' + str(adjacencies[0]) + '<br>label: ' + str(node_labels[node]) + '<br>property: ' + str(
            node_property[node]))
        node_trace['text'] += tuple([node_info])

        print(node / len(G.nodes) * 100, end='\r')

    pio.renderers.default = "browser"
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network Graph the Grid',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.show()
    return print('check it out 2d')


def show_me_graph_property_3d(dataset, property_number, show_zero_facies=True):
    '''
    3d visualization of a graph
    
    Args: graph of pytorch geometrical type
          property_number - if you want to see poro, so chose 0, if you want to see perm chose 1. it depends on dataset
          show_zero_facies - you can visualize label 0 (true) or not (false)
    
    '''
    if show_zero_facies == True:
        flag = 1
    else:
        flag = 0

    G = to_networkx(dataset)
    pos = nx.layout.spectral_layout(G, dim=3)
    pos_arr = np.asarray(list(pos.values()))
    node_labels = dataset.label[list(G.nodes)].cpu().detach().numpy()
    node_property = dataset.x[:, property_number][list(G.nodes)].cpu().detach().numpy()
    for i in torch.where(dataset['label'] == -1)[0].cpu().detach().numpy():
        G.remove_node(i)
        # Create Edges
    edge_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    x0 = pos_arr[np.asarray(list(G.edges()))[:, 0]][:, 0]
    y0 = pos_arr[np.asarray(list(G.edges()))[:, 0]][:, 1]
    z0 = pos_arr[np.asarray(list(G.edges()))[:, 0]][:, 2]

    x1 = pos_arr[np.asarray(list(G.edges()))[:, 1]][:, 0]
    y1 = pos_arr[np.asarray(list(G.edges()))[:, 1]][:, 1]
    z1 = pos_arr[np.asarray(list(G.edges()))[:, 1]][:, 2]

    Nones = np.full(len(x0), None)

    edge_trace['x'] = tuple(np.asarray(tuple(zip(x0, x1, Nones))).flatten())
    edge_trace['y'] = tuple(np.asarray(tuple(zip(y0, y1, Nones))).flatten())
    edge_trace['z'] = tuple(np.asarray(tuple(zip(z0, z1, Nones))).flatten())

    # create nodes
    node_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='bluered',
            reversescale=True,
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='Node Property',
                xanchor='left',
                titleside='right',
            ),
            line=dict(width=2)))

    node_trace['x'] = pos_arr[:, 0]
    node_trace['y'] = pos_arr[:, 1]
    node_trace['z'] = pos_arr[:, 2]

    # add color to node points
    node_trace['marker']['color'] = node_property
    sizes = np.around(node_labels + flag) * 10
    sizes[sizes < 0] = 0
    node_trace['marker']['size'] = sizes

    for node, adjacencies in enumerate(G.adjacency()):
        node_info = ('Name: ' + str(adjacencies[0]) + '<br>label: ' + str(node_labels[node]) + '<br>property: ' + str(
            node_property[node]))
        node_trace['text'] += tuple([node_info])

        print(node / len(G.nodes) * 100, end='\r')

    # pio.renderers.default = "browser"

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network Graph the Grid',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # fig.show()

    # histogram part

    hist_data = np.stack((node_property, node_labels), axis=1)
    condition = hist_data != 0
    clean_hist_data = hist_data[condition[:, 1]]

    fig1 = go.Histogram(x=clean_hist_data[:, 0], xbins=dict(start=np.min(clean_hist_data[:, 0]), size=(np.max(
        clean_hist_data[:, 0]) - np.min(clean_hist_data[:, 0])) / 50, end=np.max(clean_hist_data[:, 0])),
                        marker=dict(color='rgb(0, 0, 100)'))

    fig.update_layout(xaxis_title='mean:' + str(round(np.mean(clean_hist_data[:, 0]), 4))
                                  + '  std:' + str(round(np.std(clean_hist_data[:, 0]), 4)))

    fig.add_traces(fig1)

    # fig.show()
    return fig
    # py.plot(fig, filename = 'graph', auto_open=True)
    # fig.write_html("file.html")
    # return print('check it out 3d')


def show_me_hidden_space(zs, labels, dataset_list, number_of_graph):
    '''
    Args:
    zs - hidden space parameters
    labels - all labels of a dataset
    dataset_list - dataset itself...normalized
    number_of_graph - what graph to show
    '''
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # %matplotlib inline

    global_z = np.empty(shape=[0, 2])
    global_z_0 = []
    global_z_1 = []
    labels_z = []

    for i in tqdm(zs):
        global_z_0.append(i[:][:, 0])
        global_z_1.append(i[:][:, 1])

    global_z_0 = np.concatenate((global_z_0), axis=0)
    global_z_1 = np.concatenate((global_z_1), axis=0)
    labels_z = np.concatenate((labels), axis=0)
    global_z = np.vstack((global_z_0, global_z_1, labels_z))

    plt.subplots(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.xlabel('graph_hidden_space')
    plt.scatter(zs[number_of_graph][:, 0], zs[number_of_graph][:, 1], c=dataset_list[number_of_graph].label.cpu())
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.xlabel('graph_real_features')
    plt.scatter(dataset_list[number_of_graph].x[:, 0].cpu(), dataset_list[number_of_graph].x[:, 1].cpu(),
                c=dataset_list[number_of_graph].label.cpu())
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.xlabel('dataset_hidden_space')
    plt.scatter(global_z[0], global_z[1], c=labels_z)
    plt.colorbar()

    plt.show()


def training_process(trues, preds, path):
    '''
    Visualization of the training process through tensorboadr
    Shows true and predicted distributions for properties (limited by 2 properties)
    Args:
    trues - true distributions
    preds - predicted distributions
    path - where to store figures
    '''

    all_true = np.array(trues)
    all_pred = np.array(preds)

    all_pred = np.reshape(all_pred, (all_true.shape))

    all_true = all_true[:, 0]
    all_pred = all_pred[:, 0]

    idx = all_true[..., 0] > 0

    all_true_0 = all_true[..., 0][idx]

    all_true_1 = all_true[..., 1][idx]

    all_pred_0 = all_pred[..., 0][idx]

    all_pred_1 = all_pred[..., 1][idx]

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    canvas = FigureCanvas(fig)
    axs[0].hist(all_true_0.flatten(), 100, alpha=0.5, label='true_poro')
    axs[0].hist(all_pred_0.flatten(), 100, alpha=0.5, label='pred_poro')
    axs[0].legend(loc='upper right')

    axs[1].hist(all_true_1.flatten(), 100, alpha=0.5, label='true_perm')
    axs[1].hist(all_pred_1.flatten(), 100, alpha=0.5, label='pred_perm')
    axs[1].legend(loc='upper right')
    fig.savefig(path + '.png')

    return fig
