import classGridDataset
from torchvision import transforms
import useful_custom_functions
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


def get_dset(path, path_labels, path_prefix):
    dataset = classGridDataset.GridDataset(path_prefix + path,
                                           path_prefix + path_labels, clear_labels=True,
                                           transform=transforms.Compose([classGridDataset.GridToGraph_6(16, 12, 10)]),
                                           name='SRM_GridDataset')
    return dataset


def get_multi_dataset(path, path_labels, clear, path_prefix):
    dataset = classGridDataset.MultyProps_GridDataset(path,
                                                      path_prefix + path_labels, clear_labels=True,
                                                      transform=transforms.Compose(
                                                          [classGridDataset.GridToGraph_6(16, 12, 10, clear=clear)]),
                                                      name='SRM_GridDataset')
    return dataset


def get_dataset(name, tensor_path='../DATA/processed_dir/Channels_1500_tensor.pt',
                device=torch.device("cuda:0" if torch.cuda.is_available()
                                    else "cpu")):
    dataset_perm_train = get_dset("Permeability.csv", "Facies.csv", name)
    dataset_poro_train = get_dset("Porosity.csv", "Facies.csv", name)

    useful_custom_functions.make_me_properties_tensor(dataset_poro_train, dataset_perm_train, file_name=tensor_path)
    dataset_train = get_multi_dataset(tensor_path, "Facies.csv", True, name)
    dataset_list = []

    for d in tqdm(dataset_train):
        dataset_list.append(d.to(device))

    return dataset_list


def transform_dataset(dataset_list):
    poro = []
    perm = []

    for d in dataset_list:
        poro.append(d.x[:, 0].cpu())
        perm.append(d.x[:, 1].cpu())

    poro = np.concatenate((poro), axis=0)
    perm = np.concatenate((perm), axis=0)
    transformer, perm_normalized = useful_custom_functions.from_lognorm_to_norm(perm)

    max0 = np.amax(poro, axis=0)

    max0 = torch.tensor(max0)

    for i in range(len(dataset_list)):
        dataset_list[i].x[:, 0] /= max0
        tmp = transformer.transform(dataset_list[i].x[:, 1].reshape(-1, 1).cpu())  # normalization of every graph
        dataset_list[i].x[:, 1] = torch.tensor(tmp[:, 0])

    return dataset_list, transformer, max0


def get_dataset_item(name, idx, tensor_path='../DATA/processed_dir/Channels_1500_tensor.pt',
                     device=torch.device("cuda:0" if torch.cuda.is_available()
                                         else "cpu"),
                     preload=False):
    dataset_perm_train = get_dset("Permeability.csv", "Facies.csv", name)
    dataset_poro_train = get_dset("Porosity.csv", "Facies.csv", name)
    if not preload:
        useful_custom_functions.make_me_properties_tensor(dataset_poro_train, dataset_perm_train, file_name=tensor_path)

    dataset_train = get_multi_dataset(tensor_path, "Facies.csv", True, name)
    return dataset_train[idx].to(device)


def get_transformer(name, tensor_path='../DATA/processed_dir/Channels_1500_tensor.pt',
                    device=torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu"),
                    dataset_list=None):
    if dataset_list is None:
        dataset_perm_train = get_dset("Permeability.csv", "Facies.csv", name)
        dataset_poro_train = get_dset("Porosity.csv", "Facies.csv", name)

        useful_custom_functions.make_me_properties_tensor(dataset_poro_train, dataset_perm_train, file_name=tensor_path)
        dataset_train = get_multi_dataset(tensor_path, "Facies.csv", True, name)
        dataset_list = []

        for d in tqdm(dataset_train):
            dataset_list.append(d.to(device))

    poro = []
    perm = []

    for d in dataset_list:
        poro.append(d.x[:, 0].cpu())
        perm.append(d.x[:, 1].cpu())
    perm = np.concatenate((perm), axis=0)
    transformer, perm_normalized = useful_custom_functions.from_lognorm_to_norm(perm)

    return transformer


def OneDim_to_ThreeDim_Converter(i_max, j_max, k_max, One_Dim):
    # приводим массив к 3D размерности грида.
    Three_Dim = np.array(np.zeros((i_max, j_max, k_max)))
    i = 0
    j = 0
    k = 0
    z = 0

    while k <= k_max - 1:
        while j <= j_max - 1:
            while i <= i_max - 1:
                Three_Dim[i, j, k] = One_Dim[z]
                i += 1
                z += 1
            i = 0
            j += 1
        j = 0
        k += 1

    return Three_Dim


def unnormalize_graph(norm_graph, max0, transformer):
    pred_0 = torch.mul(norm_graph[:, 0], max0)  # unnormalize poro
    pred_1 = norm_graph[:, 1]  # *(max1-min1)+min1
    pred_1 = transformer.inverse_transform(pred_1.cpu().detach().numpy().reshape(-1, 1))
    pred_1 = torch.tensor(pred_1).to(0)
    pred_1 = pred_1[:][:, 0]

    graph = torch.stack((pred_0, pred_1), dim=1, out=None)

    return graph


def production_to_xls(data, cp, ci, NUM, path_prefix):
    """
    write matlab production data to xls

    Args:
    data - production data of wells from matlab
    cp - production wells
    ci - injection wells
    NUM - number of iteration
    """
    cumulative_oil = []

    for i in range(len(ci['points'])):
        name = f'I{i + 1}'
        df = pd.DataFrame(columns=data[i].keys())
        for key in data[i]:
            if key == "qOr" or key == "qWr":
                df[key] = data[i][key] * -86400  # convert from Si(m3/sec) to Field (m3/day)
            else:
                df[key] = data[i][key]
        df.to_excel(path_prefix + 'wells/' + str(NUM) + '_' + name + '.xlsx')

    for i in range(len(cp['points'])):
        name = f'P{i + 1}'
        df = pd.DataFrame(columns=data[i + len(ci['points'])].keys())
        for key in data[i + len(ci['points'])]:
            if key == "qOr" or key == "qWr":
                df[key] = data[i + len(ci['points'])][key] * -86400  # convert from Si(m3/sec) to Field (m3/day)
            else:
                df[key] = data[i + len(ci['points'])][key]
        cumulative_oil = np.append(cumulative_oil, [df["qOr"].sum()])
        df.to_excel(path_prefix + 'wells/' + str(NUM) + '_' + name + '.xlsx')
    print('you can find your xls file here:', path_prefix + 'wells/')
    return cumulative_oil
