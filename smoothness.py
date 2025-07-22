import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def smoothness_factor_simple(zs, trainer):
    x_last = None
    z_last = None
    derivatives = []
    with torch.no_grad():
        for z in zs:
            z = torch.tensor(np.reshape(z, (1, 30, 1))).to('cuda')
            x, _ = trainer.model.decode(z, trainer.edge_index, 1)
            if x_last is None:
                x_last = x
                z_last = z
                continue
            else:
                derivatives.append((x_last - x) / torch.sum((z_last - z)**2))
            x_last = x
            z_last = z
        second_derivatives = []
        for i in range(1, len(derivatives)):
            second_derivatives.append((derivatives[i] - derivatives[i - 1]).cpu().numpy())
        return second_derivatives


def jacobian(y, x, num_nodes, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(num_nodes):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(num_nodes, 30)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def hessian_diagonal(y, x, num_nodes):
    jac = jacobian(y, x, num_nodes, create_graph=True)
    hess = []
    for i in range(num_nodes):
        for j in range(jac.shape[1]):
            hess.append(grad(jac[i, j], x, retain_graph=True)[0])
    return torch.stack(hess).detach().cpu().numpy()


def plot_smoothness(derivatives):
    d = {}
    for i in range(len(derivatives)):
        d[i] = derivatives[i].flatten()
    plt.boxplot(d.values())
    plt.show()


def smoothness_factor_hard(zs, trainer, num_nodes=10):
    second_derivatives = []
    for z in tqdm(zs):
        z = torch.tensor(np.reshape(z, (1, 30, 1)), requires_grad=True).to('cuda')
        x, _ = trainer.model.decode(z, trainer.edge_index, 1)
        second_derivatives.append(hessian_diagonal(x, z, num_nodes))
    return second_derivatives
