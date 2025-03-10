import torch
from sklearn.preprocessing import PowerTransformer
import numpy as np  # linear algebra


def make_me_properties_tensor(*args, file_name='tensor.pt'):
    '''
    Args:
        * args - graph datasets (class GridDataset) of properties which shoul be gathered into one graph of various properties

    Return:
           file "Tensor.pt" for class MultyProps_GridDataset.    
    '''
    number_of_properties = len(args)
    number_of_models = len(args[0])
    number_of_nodes = args[0][0].num_nodes

    dataset_props_x = torch.zeros([number_of_models, number_of_nodes, number_of_properties],
                                  dtype=torch.float)  # create empty tensor for properties

    i = 0
    for i in range(len(args[0])):
        tuple_ = ((),)
        for j in range(len(args)):
            tuple_ = tuple_ + ((args[j][i]['x']),)
        j = 0
        print(i, end='\r')
        dataset_props_x[i] = torch.cat((tuple_[1:]), 1)
        # print(dataset_props_x[i])

    torch.save(dataset_props_x, file_name)
    print('you can find the tensor of properties in this file: ', file_name)


def from_lognorm_to_norm(data_in):
    data_out = []

    yj = PowerTransformer(method='yeo-johnson')

    condition = (data_in > 0)
    data_in = np.extract(condition, data_in)

    transformer = yj.fit(data_in.reshape(-1, 1))
    data_out = transformer.transform(data_in.reshape(-1, 1))

    return (transformer, data_out)
