import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from distributions import log_normal_diag, log_normal_standard
from functools import reduce
from operator import __mul__

from dgm import ELBO_ffjord
# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps # numerical logs

def tensorify(device=None, *args) -> tuple:
    return tuple(arg.to(device) if isinstance(arg, torch.Tensor) else torch.tensor(arg, device=device) for arg in args)

def eval_logp_normal(x, mean = 0., var = 1., ndim = 1):
    mean = tensorify(x.device, mean)[0]
    var = tensorify(x.device, var)[0].expand(x.shape)
    if ndim == 0:
        x = x.unsqueeze(-1); mean = mean.unsqueeze(-1); var = var.unsqueeze(-1)
        ndim = 1
    reduce_dims = tuple(range(-1, -ndim-1, -1))
    quads = ((x-mean)**2 / var).sum(dim=reduce_dims)
    log_det = var.log().sum(dim=reduce_dims)
    numel = reduce(__mul__, x.shape[-ndim:])
    return -.5 * (quads + log_det + numel * math.log(2*math.pi))

class VRNN(nn.Module):
    def __init__(self, args):
        super(VRNN,self).__init__()
        self.x_dim = args.x_dim
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.input_size=args.input_size
        self.n_layers = args.n_layers
        self.set_p_val = args.set_p_val
        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(args.x_dim, args.h_dim),
            nn.ReLU(),
            nn.Linear(args.h_dim, args.h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(args.z_dim, args.h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(args.h_dim + args.h_dim,args.h_dim),
            nn.ReLU(),
            nn.Linear(args.h_dim, args.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(args.h_dim, args.z_dim)
        #print('enc_mean shape',self.enc_mean.shape)
        self.enc_std = nn.Sequential(
            nn.Linear(args.h_dim, args.z_dim),
            nn.Softplus())
        #print('self.enc_std.shape is',self.enc_std.shape)
        #self.enc_std =self.get_p_val()
        #print(self.enc_std.shape)

        #prior
        self.prior = nn.Sequential(
            nn.Linear(args.h_dim, args.h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(args.h_dim, args.z_dim)
        self.prior_std = nn.Sequential(nn.Linear(args.h_dim, args.z_dim),nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(args.h_dim + args.h_dim, args.h_dim),
            nn.ReLU(),
            nn.Linear(args.h_dim, args.h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(args.h_dim, args.x_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(args.h_dim, args.x_dim),
            nn.Sigmoid())
        #recurrence
        self.rnn = nn.GRU(args.h_dim + args.h_dim, args.h_dim, args.n_layers, args.bias)

        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

    def forward(self, x):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss,nll_loss = 0,0
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)
        #ELBO_ffjord(n_mc_px, eval_logprior, draw_prior, eval_logp, draw_p, *,    eval_logq = None, draw_q = None,
        #    draw_q0 = None, eval_z1eps_logqt = None, eval_z1eps = None)
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            #print('phi_x_t is',phi_x_t,phi_x_t.shape)
 #           ELBO = dgm.ELBO_ffjord(args.n_mc_px, **model_args)
            #loss, kld_loss, nll_loss = self.getlosses(x)
            #print(kld_loss, nll_loss)
            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            #print('enc_t.shape is',enc_t.shape)
            enc_mean_t = self.enc_mean(enc_t)
            #print('enc_mean_t is',enc_mean_t.shape)
            enc_std_t = self.enc_std(enc_t)
            #prior
            prior_t = self.prior(h[-1])
            #print('prior_t is',prior_t,prior_t.shape)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            #print('prior_mean_t is',prior_mean_t.shape)
            #print('prior_t std is',prior_std_t.shape)
            #sampling and reparameterization
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            #print('z_t.shape is',z_t.shape)
            phi_z_t = self.phi_z(z_t)
            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)
            #print('dec_mean_t is',dec_mean_t.shape)
            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            #computing losses
            #kle_loss,nll_loss=self.getlosses(x,)
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            #print(x[t])
            #print(dec_mean_t.shape)
            #print(dec_std_t.shape)
            #print(x[t].shape)
            #print(kld_loss)
            #print(nll_loss)
            #nll_loss += self._nll_bernoulli(dec_mean_t, x[t])
            #all_enc_std.append(enc_std_t)
            #all_enc_mean.append(enc_mean_t)
            #all_dec_mean.append(dec_mean_t)
            #all_dec_std.append(dec_std_t)
        return kld_loss, nll_loss#, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std)

    def sample(self, seq_len):
        sample = torch.zeros(seq_len, self.x_dim, device=device)
        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):
            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            #sampling and reparameterization
            z_t = self.reparameterize(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)
            phi_x_t = self.phi_x(dec_mean_t)
            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            sample[t] = dec_mean_t.data
        return sample

    def get_p_val(self):
        set_p_val = self.set_p_val
        if bool(set_p_val): set_p_val = torch.tensor(set_p_val)
        if not bool(set_p_val) or (set_p_val <= 0).any().item():
            p_val = nn.Sequential(
                nn.Linear(self.dims_z2h[-1], self.dim_x),
                nn.Softplus(),
                ViewLayer(self.input_size),
            )
        else:
            p_val_tensor = set_p_val.expand(self.input_size)
            p_val = lambda x: p_val_tensor.to(x)
        return p_val

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        print(eps.shape)
        return eps.mul(std).add_(mean)

    def reparameterize(self, mu, var):
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().to(mu)
        eps = Variable(eps)
        if hasattr(self, 'vae_clamp'): eps.data.clamp_(-self.vae_clamp, self.vae_clamp)
        z = eps.mul(std).add_(mu)
        return z



    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sigmoid(torch.sum(torch.log(std + EPS) + torch.log(torch.tensor(2*torch.pi))/2 + (x - mean).pow(2)/(2*std.pow(2))))

    def getlosses(self, x, kl_beta = 1.):
        reconstruction_function = nn.MSELoss(size_average=False)
        batch_size = x.size(0)
        self.input_size=batch_size
        x_mean, z_mu, z_var, ldj, z_0, z_k = self.draw_q0(x)
        bce = reconstruction_function(x_mean, x)
        log_p_zk = log_normal_standard(z_k, dim=0)
        # ln q(z_0)  (not averaged)
        log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=0)
        print(log_p_zk,log_q_z0)
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        summed_logs = torch.sum(log_q_z0 - log_p_zk)
        # sum over batches
        summed_ldj = torch.sum(ldj)
        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        print(summed_logs,summed_ldj)
        kl = (summed_logs - summed_ldj)
        print(kl)
        loss = bce + kl_beta * kl
        #loss /= float(batch_size)
        #bce /= float(batch_size)
        #kl /= float(batch_size)
        # print(bce.data.item(), kl.data.item())
        return loss, bce, kl

    def draw_q0(self, x, n_mc = 6):
        # print(x.size())
        batch_size = x.shape[:-self.input_size]
        eps = torch.randn((n_mc,) + batch_size + (self.z_dim,), device=x.device)
        if hasattr(self, 'vae_clamp'):
            eps.data.clamp_(-self.vae_clamp, self.vae_clamp)
        return eps
    def draw_q(self, x, n_mc = 1):
        return self.eval_z1eps(x, self.draw_q0(x, n_mc))

class ViewLayer(nn.Module):
    def __init__(self, shape):
        self.shape = torch.Size(shape)

    def forward(self, x):
        if x.ndim < 2: return x.view(*self.shape)
        else: return x.view(-1, *self.shape)