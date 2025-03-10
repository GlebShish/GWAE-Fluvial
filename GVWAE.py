import math
from tqdm import tqdm
import copy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch
from sklearn.cluster import KMeans
from brownian_motion import brownian_motion_sample, log_bm_krn
import numpy as np
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import visual_tools
from itertools import chain
from torch.distributions import Normal
from basic_layers import RBF, PosLinear, Reciprocal, Sqrt
import nnj
from nnj import GraphConv


class Adversary(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=10):
        super(Adversary, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 1),  # B,   1
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def sample_z(sigma=None, template=None):
    if template is not None:
        z = sigma * Variable(template.data.new(template.size()).normal_())
        return z
    else:
        return None


def mmd(z_tilde, z, z_var=1):
    r"""Calculate maximum mean discrepancy described in the WAE paper.
    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        z (Tensor): samples from prior distributions. same shape with z_tilde.
        z_var (Number): scalar variance of isotropic gaussian prior P(Z).
    """
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(1)

    out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n * (n - 1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n * (n - 1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n * n).mul(2)

    return out


def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2 * z_dim * z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C / (1e-9 + C + (z11 - z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def log_density_igaussian(z, z_var):
    """Calculate log density of zero-mean isotropic gaussian distribution given z and z_var."""
    assert z.ndimension() == 2
    assert z_var > 0

    z_dim = z.size(1)

    return -(z_dim / 2) * math.log(2 * math.pi * z_var) + z.pow(2).sum(1).div(-2 * z_var)


class RVAE(nn.Module):
    def __init__(self, channels, nv, nz, edge_index, num_centers, rbf_beta, rec_b, middle_layer_size, middle_channels,
                 batch_size=1):
        super(RVAE, self).__init__()
        self.channels = channels
        self.num_centers = num_centers
        self.nv = nv
        self._mean_warmup = True
        self.nz = nz
        self.encoder = nnj.Sequential(GraphConv(self.channels, 2, edge_index, batch_size, aggr='mean'), nnj.ELU(),
                                      GraphConv(2, 1, edge_index, batch_size, aggr='mean'), nnj.ELU())
        self.q_mu = nn.Sequential(
            nn.Linear(self.nv, self.nz)
        )
        self.q_t = nn.Sequential(
            nn.Linear(self.nv, self.nz),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.)
        )
        self.dummy_pmu = nnj.Sequential(nnj.Linear(self.nz, middle_layer_size), nnj.ELU(),
                                        nnj.Linear(middle_layer_size, self.nv), nnj.ELU(),
                                        GraphConv(1, middle_channels, edge_index, batch_size, aggr='mean'), nnj.ELU(),
                                        GraphConv(middle_channels, self.channels, edge_index, batch_size, aggr='mean'))

        self.p_mu = nnj.Sequential(nnj.Linear(self.nz, middle_layer_size), nnj.ELU(),
                                   nnj.Linear(middle_layer_size, self.nv), nnj.ELU(),
                                   GraphConv(1, middle_channels, edge_index, batch_size, aggr='mean'), nnj.ELU(),
                                   GraphConv(middle_channels, self.channels, edge_index, batch_size, aggr='mean'))

        self.p_sigma = nnj.Sequential(
            nnj.RBF(self.nz, num_points=num_centers, beta=rbf_beta),
            nnj.PosLinear(num_centers, self.nv * self.channels, bias=False),
            nnj.Reciprocal(b=rec_b),
            nnj.Sqrt()
        )
        self._latent_codes = None
        self.pr_means = torch.nn.Parameter(torch.zeros(self.nz), requires_grad=True)
        self.pr_t = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def change_batch_size(self, edge_index):
        for module in self.dummy_pmu._modules:
            if type(module) == GraphConv:
                module.change_batch_size(edge_index)
        for module in self.p_mu._modules:
            if type(module) == GraphConv:
                module.change_batch_size(edge_index)
        for module in self.encoder._modules:
            if type(module) == GraphConv:
                module.change_batch_size(edge_index)

    def encode(self, x, batch_size):
        x = self.encoder(x)
        x = x.view(batch_size, self.nv)

        q_mu = self.q_mu(x)
        q_t = self.q_t(x)

        eps = torch.randn_like(q_mu)

        # reparameterize
        z = (q_mu + q_t.sqrt() * eps).view(-1, self.nz)

        return z, q_mu, q_t

    def _update_latent_codes(self, data_loader, batch_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = []
        print("updating latent codes")
        for _, data in tqdm(enumerate(data_loader)):
            # dim1, dim2 = data.shape[-2], data.shape[-1]
            z, _, _ = self.encode(data.x.to(device), batch_size)
            codes.append(z)
        self._latent_codes = torch.cat(codes, dim=0).view(-1, self.nz)

    def _update_RBF_centers(self, beta=None):

        print("Updating RBF centers")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kmeans = KMeans(n_clusters=self.num_centers, verbose=1)
        kmeans.fit(self._latent_codes.detach().cpu().numpy())
        self.p_sigma._modules['0'].points.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
        self.p_sigma._modules['0'].beta = beta

    def _initialize_prior_means(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(self._latent_codes.detach().cpu().numpy())
        self.pr_means.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)

    def decode(self, z, jacobian):
        if self._mean_warmup:
            if jacobian:
                mu, J_mu = self.p_mu(z, jacobian)
                sigma, J_sigma = self.p_sigma(z, jacobian)

                J_mu = torch.einsum("bij,bkj->bij", J_mu, J_mu)
                # J_sigma = torch.einsum("bij,bkj->bij", J_sigma, J_sigma)

                return mu, sigma, J_mu
            else:
                mu = self.p_mu(z, jacobian)
                sigma = self.p_sigma(z, jacobian)

                return mu, sigma
        else:
            if jacobian:
                sigma, J_sigma = self.p_sigma(z, jacobian)
                mu, J_mu = self.p_mu(z, jacobian)

                # Get quadratic forms of Jacobians
                J_mu = torch.einsum("bij,bkj->bij", J_mu, J_mu)
                J_sigma = torch.einsum("bij,bkj->bij", J_sigma, J_sigma)
                # Compute metric
                G = J_mu + J_sigma

                return mu, sigma, G
            else:
                mu = self.p_mu(z)
                sigma = self.p_sigma(z)
                return mu, sigma

    def sample(self, num_steps, num_samples, keep_last):
        """Generate samples from a Brownian motion on the manifold.

        Params:
            num_steps:      int - the number of discretized steps
                            of the simulated Brownian motion
            num_samples:    int - the number of returned samples
            keep_last:      bool - if true keeps only the last step
                            of the Brownian motion
            device:         str - "cuda" or "cpu"
        """

        self.eval()
        samples = brownian_motion_sample(num_steps, num_samples, self.latent_dim, self.pr_t, self.pr_means.data, self)

        if keep_last:
            if samples.dim() == 3:
                samples = samples[-1, :, :]
            else:
                samples = samples[-1, :]
        x = self.p_mu(samples)

        return x, samples

    def metric(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # print('STARTING')
        _, J_mu = self.p_mu(z, True)
        J_mu = torch.einsum("bji,bjk->bik", J_mu, J_mu)
        _, J_sigma = self.p_sigma(z, True)
        J_sigma = torch.einsum("bji,bjk->bik", J_sigma, J_sigma)

        return J_mu + J_sigma

    def forward(self, x, batch_size, jacobian=False):
        z, q_mu, q_var = self.encode(x, batch_size)
        if jacobian:
            p_mu, p_sigma, G = self.decode(z, jacobian)
            p_sigma = p_sigma.reshape(1, -1)
            return p_mu, p_sigma, z, q_mu, q_var, G
        else:
            p_mu, p_sigma = self.decode(z, jacobian)
            p_sigma = p_sigma.reshape(1, -1)
            # p_sigma = torch.reshape(p_sigma, (-1,self.channels))
            return p_mu, p_sigma, z, q_mu, q_var


def elbo_rvae(data, p_mu, p_sigma, z, q_mu, q_t, model, beta):
    """

    :param data:
    :param p_mu:
    :param p_sigma:
    :param z:
    :param q_mu:
    :param q_t:
    :param model:
    :param beta: KL loss parameter
    :return:
    """
    data = data.reshape(p_mu.shape)
    if model._mean_warmup:
        norm = Normal(p_mu, p_sigma)
        norm_log_prob = norm.log_prob(data)
        nlps = norm_log_prob.sum(-1).sum(-1)
        nlpsm = -nlps.mean()
        return nlpsm, torch.zeros(1), torch.zeros(1)
    else:

        p_sigma = p_sigma.reshape(p_mu.shape)
        pr_mu, pr_t = model.pr_means, model.pr_t

        log_pxz = Normal(p_mu, p_sigma).log_prob(data).mean(-1)
        log_qzx = log_bm_krn(z, q_mu, q_t, model)
        log_pz = log_bm_krn(z, pr_mu.expand_as(z), pr_t, model)

        KL = log_qzx - log_pz
        return (-log_pxz + beta * KL.abs()).mean(), -log_pxz.mean(), KL.mean()


class TrainerRBF:
    def __init__(self, batch_size=1, nv=1919, nz=500, model_loss='mmd', gamma=0, channels=2,
                 num_centers=None, warmup_learning_rate=1e-4, sigma_learning_rate=1e-4,
                 edge_index=None, rbf_beta=0.01, rec_b=1e-9, writer=None, middle_layer_size=100, middle_channels=2):
        """

        :param batch_size: batch_size for training and inference
        :param nv: number of vertices in a graph
        :param nz: size of latent codes
        :param model_loss: extra_loss 'mmd', 'gan' or 'kl
        :param gamma: extra loss multiplier
        :param channels: number of values in each vertex
        :param num_centers: number of cluster centers for RBF
        :param warmup_learning_rate: learning rate for mu optimization stage
        :param sigma_learning_rate: learning rate for sigma optimization stage
        :param edge_index: edge_index of graph
        :param rbf_beta:
        :param rec_b:
        :param writer: TensorBoard writer
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.nv = nv
        self.nz = nz

        self.edge_index = edge_index
        self.batch_size = batch_size
        self.num_centers = num_centers

        self.switch = True
        self.writer = writer
        self.model_loss = model_loss

        assert self.model_loss in ['gan', 'mmd', 'kl']

        if self.model_loss == 'gan':
            self.D = Adversary(self.nz).to(self.device)
            self.optim_D = torch.optim.Adam(self.D.parameters(), lr=1e-4,
                                            betas=(0.5, 0.999))

        self.gamma = gamma
        self.channels = channels

        self.model = RVAE(channels, nv, nz, edge_index, num_centers,
                          rbf_beta, rec_b, middle_layer_size,
                          middle_channels, batch_size).to('cuda')
        self.warmup_optimizer = torch.optim.Adam(
            chain(
                self.model.encoder.parameters(),
                self.model.q_mu.parameters(),
                self.model.q_t.parameters(),
                self.model.p_mu.parameters()),
            lr=warmup_learning_rate
        )

        self.sigma_optimizer = torch.optim.Adam(
            chain(
                self.model.p_sigma.parameters(),
                [self.model.pr_means, self.model.pr_t]),
            lr=sigma_learning_rate
        )

    def train(self, mu_epochs, sigma_epochs, train_loader, path_prefix):
        wandb_run = wandb.init(project='rvae', entity='hits')
        wandb_run.watch(self.model)

        scheduler = CosineAnnealingLR(self.warmup_optimizer, len(train_loader) * mu_epochs // self.batch_size,
                                      eta_min=0, last_epoch=-1)
        # wandb_run.watch(self.model, log='all', log_freq=1000)
        batch_size = self.batch_size
        iteration = 0
        self.model.train()
        for epoch in tqdm(range(mu_epochs)):
            preds = []
            trues = []
            for batch in tqdm(train_loader):
                # facies = batch.label.float().to('cuda').reshape(-1,1)
                batch = batch.x.float()
                # batch = torch.cat([batch, facies], dim=1)
                extra_loss, mse_loss, pred, z = self.train_on_batch(batch, True)
                if wandb_run is not None and iteration % 30 == 0:
                    wandb_run.log({'ELBO': mse_loss, self.model_loss: extra_loss})
                    wandb_run.log({'warmup_lr': self.warmup_optimizer.param_groups[0]['lr']})
                preds.append(pred.cpu().detach().numpy())
                trues.append(np.reshape(batch.cpu().detach().numpy(), (batch_size, self.nv, self.channels)))
                iteration += 1
                scheduler.step()
            fig = visual_tools.training_process(trues, preds, path_prefix + 'figs/' + str(epoch))
            if wandb_run is not None:
                wandb_run.log({'z_dims_2': wandb.Image(fig)})

        batch = next(iter(train_loader))
        pred, z = self.test_on_batch(batch.x.float())
        pred = torch.reshape(pred, (self.batch_size, self.nv, -1))

        fake_batch1 = copy.deepcopy(batch)
        fake_batch2 = copy.deepcopy(batch)
        fake_batch2.x = pred[0]

        fake_batch1.x = fake_batch1.x[:len(fake_batch1.x) // self.batch_size]

        fake_batch1.edge_index = fake_batch1.edge_index[:, :fake_batch1.edge_index.shape[1] // self.batch_size]
        fake_batch2.edge_index = fake_batch2.edge_index[:, :fake_batch2.edge_index.shape[1] // self.batch_size]
        fig1 = visual_tools.show_me_graph_property_3d(fake_batch1, 0)
        fig2 = visual_tools.show_me_graph_property_3d(fake_batch2, 0)

        if wandb_run is not None:
            wandb_run.log({'True': fig1, 'Pred': fig2})

        self.model._update_latent_codes(train_loader, batch_size)
        self.model._update_RBF_centers(beta=0.01)
        self.switch = False
        self.model._mean_warmup = False
        self.model._initialize_prior_means()
        self.model.dummy_pmu.load_state_dict(self.model.p_mu.state_dict())
        print("Starting sigma optimization...")
        for epoch in tqdm(range(sigma_epochs)):
            preds = []
            trues = []
            for batch in tqdm(train_loader):
                # facies = batch.label.float().to('cuda')
                batch = batch.x.float().to('cuda')
                # batch = torch.concatenate([batch, facies])
                extra_loss, mse_loss, pred, z = self.train_on_batch(batch, False)
                if wandb_run is not None:
                    wandb_run.log({'ELBO': mse_loss, self.model_loss: extra_loss})
                preds.append(pred.cpu().detach().numpy())
                trues.append(np.reshape(batch.cpu().detach().numpy(), (batch_size, self.nv, self.channels)))
                iteration += 1
            fig = visual_tools.training_process(trues, preds, path_prefix + 'figs/' + str(epoch + mu_epochs))
            if wandb_run is not None:
                wandb_run.log({'z_dims_2': wandb.Image(fig)})

    def train_on_batch(self, batch, warmup=True):
        self.warmup_optimizer.zero_grad()
        self.sigma_optimizer.zero_grad()

        p_mu, p_sigma, z, q_mu, q_t = self.model(batch, self.batch_size)

        prior_z = sample_z(template=z.clone().detach().view(self.batch_size, self.nz), sigma=1)

        if self.model_loss == 'mmd':
            extra_loss = mmd(z.view(self.batch_size, self.nz), prior_z, z_var=1)

        elif self.model_loss == 'kl':
            extra_loss = self.model.kl_loss()

        elif self.model_loss == 'gan':
            ones = cuda(torch.ones(self.batch_size, 1), True)
            second_ones = cuda(torch.ones(self.batch_size, 1), True)
            zeros = cuda(torch.zeros(self.batch_size, 1), True)
            z_for_d = z.clone().detach().view(self.batch_size, self.nz)
            log_p_z = log_density_igaussian(z_for_d, 1).view(-1, 1)
            D_z = self.D(z_for_d)
            D_z_tilde = self.D(prior_z)
            D_loss = F.binary_cross_entropy_with_logits(D_z + log_p_z, ones) + \
                     F.binary_cross_entropy_with_logits(D_z_tilde + log_p_z, zeros)

            self.optim_D.zero_grad()
            D_loss.backward(retain_graph=True)

            extra_loss = F.binary_cross_entropy_with_logits(D_z_tilde + log_p_z, second_ones)

        if self.switch:
            p_sigma = torch.ones(1).to(self.device)

        elbo_loss, _, _ = elbo_rvae(batch, p_mu, p_sigma, z, q_mu, q_t, self.model, 1.)
        loss = elbo_loss + extra_loss * self.gamma

        if warmup:
            loss.backward()
            self.warmup_optimizer.step()
        else:
            loss.backward()
            self.sigma_optimizer.step()

        if self.model_loss == 'gan':
            self.optim_D.step()

        return extra_loss, elbo_loss, p_mu, z

    def test_on_batch(self, batch):
        batch = batch.x.float().to('cuda')
        with torch.no_grad():
            p_mu, p_sigma, z, q_mu, q_t = self.model(batch, self.batch_size)

        return p_mu, z
