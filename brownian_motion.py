import torch
from math import pi
from torch.distributions import Normal, Categorical
from curve import connecting_geodesic


class DistSqKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, net, p0, p1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        b_sz = p0.shape[0]
        with torch.no_grad():
            with torch.enable_grad():
                crv, energy = connecting_geodesic(net, p0, p1, max_iter=12, n_nodes=5, eval_grid=16, l_rate=1e-3)
                lm0 = crv.deriv(torch.zeros(1, device=device)).view(b_sz, -1)
                lm1 = crv.deriv(torch.ones(1, device=device)).view(b_sz, -1)
                ctx.save_for_backward(lm0, lm1)
        net.p_sigma.zero_grad()
        net.dummy_pmu.zero_grad()
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.dim() == 1:
            grad_output.unsqueeze_(1)
        lm0, lm1 = ctx.saved_tensors
        return None, 2 * grad_output * lm0, 2 * grad_output * lm1


def log_bm_krn(x, y, t, model):
    """Log pdf of a Brownian motion (BM) transition kernel.

    params:
        x:      torch.tensor object - a point on the manifold
        y:      torch.tensor object - a point on the manifold,
                typically interpreted as a "mean".
        t:      float - the time for which the BM occur
        model:  nn.Module object - the model containing the embedding
                mapping to the ambient space
    """
    d = x.size(1)
    t = t.squeeze()
    _, logdet_x = model.metric(x).slogdet()
    _, logdet_y = model.metric(y).slogdet()
    log_H = (logdet_x - logdet_y) / 2
    l_sq = DistSqKL.apply(model, x, y)
    return -d / 2 * torch.log(2 * pi * t) + log_H - l_sq / (2 * t)


def brownian_motion_sample(n_steps, dim, t, init_point, model):
    """Returns the points of a discretized Brownian motion (BM)
    on a manifold (a.k.a. latent space).

    params:
        n_steps:        int - the number of time steps for which
                        the BM will run
        dim:            int - the dimensionality of the manifold/
                        latent space
        t:              float - the time for which the BM will run
        init_point:     torch.Tensor - the initial point of the BM
        model:          torch.nn.Module - the model containing the
                        embedding
    """
    if init_point is None:
        init_point = torch.zeros(dim)
    samples = [init_point.squeeze()]

    for _ in range(n_steps - 1):
        g = model.metric(samples[-1])
        cov_mat = t / n_steps * g.squeeze().inverse()
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(samples[-1], covariance_matrix=cov_mat)
        samples.append(mvn.sample().squeeze())

    return torch.stack(samples)


def multibrownian_motion_sample(k, mu, t, dim, n_steps, model):
    """Returns the points of a discretized mixture of Brownian
    motions (mBM) on a manifold (a.k.a) latent space given
    mixture probabilities, means and time lengths.

    params:
        k:          torch.tensor - mixture probabilities
        mu:         torch.tensor - the "means"/centers of
                    the BMs
        t:          float - the time for which the BM will run
        dim:        int - the dimensionality of the manifold/
                    latent space
        n_steps:    int - the number of discretized time steps
                    for which the BM will run
        model:      torch.nn.Module - the model containing the
                    embedding
    """
    idx = Categorical(probs=k).sample()
    samples = [mu[idx]]

    for _ in range(n_steps - 1):
        g = model.metric(samples[-1])
        g_inv = g.squeeze().inverse()
        cov_mat = (t[idx] / n_steps) * g_inv
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(samples[-1], covariance_matrix=cov_mat)
        samples.append(mvn.sample().squeeze())

    return torch.stack(samples)


def log_gauss_mix(x, mu, var):
    # number of components
    K = mu.shape[0]

    x_xp = x.unsqueeze(1)
    mu_xp = mu.unsqueeze(0)
    var_xp = var.unsqueeze(0)

    a = log_Normal_diag(x_xp, mu_xp, torch.log(var_xp + 1e-5), dim=2) - math.log(K)
    a_max, _ = torch.max(a, 1)  # MB x 1

    # calculte log-sum-exp
    log_mix = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))

    return log_mix


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)
