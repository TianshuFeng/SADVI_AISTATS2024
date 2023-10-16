#%% # prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multinomial import Multinomial
from scipy.interpolate import BSpline
import numpy as np
from torch.distributions import MultivariateNormal, Normal, RelaxedOneHotCategorical
import tqdm


class BVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, z_dim,
                 k: int = 3, t: list = [0.25, 0.5, 0.75], n_mcmc = 10000,
                 device = 'cpu', mnist_transform = True):
        """_summary_

        Args:
            input_dim (int): Input dimension
            hidden_dim (list): List of hidden layer dimensions
            z_dim (_type_): Latent space dimension
            k (int, optional): Degree of BSpline. Defaults to 3.
            t (list, optional): List of inner knots. boundary knots are not included and set to 0 and 1. 
                Defaults to [0.3, 0.5, 0.6].
        """        

        super(BVAE, self).__init__()
        # Initial spline basis
        
        self.device = device
        
        self.mnist_transform = mnist_transform
        
        self.k = k 
        self.t = [0]*(k+1) + t + [1]*(k+1)
        self.n_spl_basis = len(self.t) - self.k - 1
        self.n_mcmc = n_mcmc
        
        self.init_bspline_mcmc()
        self.init_bspline_penalty()
        
        # initial network layers and components
        self.z_dim = z_dim
        
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim[0])])
        self.decoder_layers = nn.ModuleList([nn.Linear(hidden_dim[0], input_dim)])
        self.decoder_layers_var = nn.ModuleList([nn.Linear(hidden_dim[0], input_dim)])
                
        if len(hidden_dim)>1:
            for i in range(len(hidden_dim)-1):
                self.encoder_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                self.decoder_layers.insert(0, nn.Linear(hidden_dim[i+1], hidden_dim[i]))
                self.decoder_layers_var.insert(0, nn.Linear(hidden_dim[i+1], 
                                                            hidden_dim[i]))
                
        self.encoder_layers.append(nn.Linear(hidden_dim[-1], (self.n_spl_basis + 2) * z_dim))
        self.decoder_layers.insert(0, nn.Linear(z_dim, hidden_dim[-1]))
        self.decoder_layers_var.insert(0, nn.Linear(z_dim, hidden_dim[-1]))

        self.to(device)
        
    def init_bspline_mcmc(self):
        self.basis_func = []
        self.basis_integral = []
        self.basis_mcmc = []
        for i in range(self.n_spl_basis):
            tmp_spl = self.bspline_basis(i=i)
            
            self.basis_func.append(tmp_spl)
            self.basis_integral.append(tmp_spl.integrate(0, 1))
            self.basis_mcmc.append(self.bspline_mcmc(tmp_spl, self.n_mcmc))
        
        self.basis_integral = np.array(self.basis_integral, dtype="float32")
        self.basis_mcmc = np.array(self.basis_mcmc, dtype="float32")
    
    
    def init_bspline_penalty(self, n_approx = 100):
        xx = np.linspace(0, 1., n_approx+1)
        UU2 = np.zeros((n_approx+1, 
                             len(self.t) - self.k - 1))
        for idx, spl in enumerate(self.basis_func):
            UU2[:,idx] = spl.derivative(2)(xx)

        self.spline_penalty_matrix = torch.tensor(0.01 * UU2.T @ UU2,
                                           device=self.device)
    
    def bspline_basis(self, i) -> BSpline:
        c = [0]*(len(self.t) - self.k - 1)
        c[i] = 1
    
        return BSpline(self.t, c, self.k, extrapolate=False)
    
    def encoder(self, x):
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if idx < len(self.encoder_layers) - 1:
                x = F.elu(x)
        return x.view(x.shape[0], x.shape[1], self.z_dim, -1) # each row corresponds to a coefs of a latent variable
    
    def latent_scaling(self, latent_vars):
        
        # With sigmoid
        coef_spl = F.softmax(latent_vars, dim = -1)  # T * batch * z_dim * n_basis   # between 0 and 1 for multinomial distribution
        basis_integral = torch.tensor(self.basis_integral.reshape(1,1,1,-1), device = self.device) # 1 * 1 * 1 * n_basis
        weights = coef_spl/basis_integral # T * batch * z_dim * n_basis  # For constructing the bpline approximation

        return coef_spl, weights
    
    def approx_sampling(self, coef_spl, temperature = 0.1):
        roc_dist = RelaxedOneHotCategorical(torch.tensor([temperature]).to(self.device),
                                            probs=coef_spl)
        
        indicator_approx = roc_dist.rsample() # T * batch * z_dim * n_basis
        
        T = indicator_approx.shape[0]
        batch_size = indicator_approx.shape[1]
        
        basis_samples = self.basis_mcmc[list(range(self.n_spl_basis)) * self.z_dim * batch_size * T, 
                                np.random.randint(0, self.basis_mcmc.shape[1], 
                                                  size=self.n_spl_basis * self.z_dim * batch_size * T)]
        basis_samples = basis_samples.reshape(T,
                                              batch_size,
                                              self.z_dim,
                                              self.n_spl_basis)  # T * batch * z_dim * n_basis  
        spl_samples = (indicator_approx * torch.tensor(basis_samples).to(self.device)).sum(-1) # T * batch * z_dim
        pdf_matrix_spl = self.pdf_matrix_bspline(basis_samples=basis_samples)
        
        pdf_approx = torch.matmul(coef_spl.unsqueeze(-2), # Turn it into T X batch * z_dim * 1 * n_basis
                                  pdf_matrix_spl  # T * batch * z_dim * n_basis * n_basis
                                  ) # T * batch * z_dim * 1 * n_basis
        pdf_approx = torch.matmul(pdf_approx, # T * batch * z_dim * 1 * n_basis
                                  indicator_approx.unsqueeze(-1)  # T * batch * z_dim * n_basis * 1
                                  )  # T * batch * z_dim * 1 * 1
        return spl_samples, pdf_approx.squeeze(-1).squeeze(-1)  # T * batch * z_dim
    
    def pdf_matrix_bspline(self, basis_samples):
        
        # To generate a matrix of pdf of bspline basis, the ij-th entry of the matrix is 
        # pdf value of i-th basis for j-th sample, b^i(z^j)
        # z^j is the realization from j-th basis
        
        pdf_matrix_spl = []
        for ind_basis in range(self.n_spl_basis):
            pdf_spl_each_basis_all_z = []
            for ind_z in range(self.n_spl_basis):
                pdf_spl_tmp = self.basis_func[ind_basis](basis_samples[...,ind_z])/self.basis_integral[ind_basis]  # T * batch * z_dim
                pdf_spl_each_basis_all_z.append(pdf_spl_tmp)
        
            pdf_spl_each_basis_all_z = np.stack(pdf_spl_each_basis_all_z, axis = -1)  # T * batch * z_dim * n_basis
            pdf_spl_each_basis_all_z = torch.tensor(pdf_spl_each_basis_all_z, device = self.device).float()  # ind_basis-th row of bspline matrix

            pdf_matrix_spl.append(pdf_spl_each_basis_all_z)
        
        res = torch.stack(pdf_matrix_spl, -2)  # combine the rows, along the second last dim (the last is column)
        
        return res  # T * batch * z_dim * n_basis * n_basis
    
    def static_sampling(self, weights):    
        # This sampling result is not differetiable and cannot be back propagated 
        
        multi_nomial = Multinomial(probs = weights)
        
        sampled_basis_index = multi_nomial.sample().argmax(dim = 2) # batch * z_dim * 1 ( each entry is a basis index)
        sampled_basis_index_flat = sampled_basis_index.view(-1).numpy()
        static_sampled_latent_value = self.basis_mcmc[sampled_basis_index_flat, 
                                              np.random.randint(0, self.basis_mcmc.shape[1], 
                                                                len(sampled_basis_index_flat))]
        
        static_sampled_latent_value = static_sampled_latent_value.reshape(sampled_basis_index.shape)  # batch * z_dim
        
        return static_sampled_latent_value # return z sample
        
    def decoder(self, z):
        
        z_var = z.clone()
        for idx, (layer_mu, layer_var) in enumerate(zip(self.decoder_layers,
                                                        self.decoder_layers_var)):
            z = layer_mu(z)
            z_var = layer_var(z_var)
            if idx < len(self.decoder_layers) - 1:
                z = F.elu(z)
                z_var = F.elu(z_var)
        return torch.sigmoid(z), torch.sigmoid(z_var) 
    
    def forward(self, x, temperature = 0.1):
        
        latent_vars = self.encoder(x)  # T X batch X z_dim X (n_basis + 2)
        
        mu = latent_vars[..., 0]
        log_var = latent_vars[..., 1]  # T X batch X z_dim
        z_std = log_var.mul(0.5).exp_()
        latent_vars = latent_vars[..., 2:]
        
        coef_spl, weights = self.latent_scaling(latent_vars=latent_vars)
        z_sample_approx, pdf_approx = self.approx_sampling(coef_spl=coef_spl,
                                                          temperature = temperature)  # both are T * batch * z_dim
        z_sample_approx = z_sample_approx * z_std + mu
        pdf_approx = pdf_approx
        recon_mean, recon_var = self.decoder(z_sample_approx)

        return recon_mean, recon_var, coef_spl, weights, z_sample_approx, pdf_approx, z_std
    
    @staticmethod
    def loss_function(x, recon_mean, recon_var, z, pdf_approx, z_std, prior,  beta = 1):
        
        z_dim = z.shape[-1]
        device = z.device
        # IWAE 
        # Find p(x|z)
        std_dec = recon_var.mul(0.5).exp_()
        px_Gz = Normal(loc=recon_mean, scale=std_dec).log_prob(x)  # T X batch_size X x_dim
        log_PxGz = torch.sum(px_Gz, -1)  # T X batch_size
        # Find p(z)
        log_Pz = prior.log_prob(z)  
        # Find q(z|x)
        log_QzGx = torch.log(pdf_approx).sum(-1)  # T X Batch
        log_loss = log_PxGz + (log_Pz - log_QzGx)*beta  
        return -torch.mean(torch.logsumexp(log_loss, 0) + torch.log(z_std.mean(0)).sum(-1))
    
    def calc_loss(self, x, T = 1, prior = None, beta = 0.05, temperature = 0.1,
                  coef_entropy_penalty = 10, coef_indi_penalty = 0,
                  coef_spline_penalty = 0):
        
        # Set T=1 to use ELBO, T>1 to use IWAE
        batch_size = x.shape[0]
        x = x.expand(T, batch_size, -1).to(self.device)
        
        recon_mean, recon_var, coef_spl, weights, \
            z_sample_approx, pdf_approx, z_std = self.forward(x, 
                                                              temperature = temperature)
        if prior is None:
            prior = MultivariateNormal(loc = torch.zeros(self.z_dim, 
                                                         device = self.device),
                                       scale_tril=torch.eye(self.z_dim, 
                                                            device = self.device))
        loss = self.loss_function(x, recon_mean, recon_var, 
                                  z_sample_approx, pdf_approx, z_std = z_std, 
                                  prior = prior,
                                  beta = beta)
        entropy_penalty, indi_penalty = self.mixture_penalties(coef_spl)
        entropy_penalty = torch.mean(torch.logsumexp(entropy_penalty, 0))
        indi_penalty = torch.mean(torch.logsumexp(indi_penalty, 0))
        # Use weights
        spline_penalty = (torch.matmul(weights.float(), self.spline_penalty_matrix.float()) * weights.float()).sum(-1)  # T X batch_size X z_dim
        spline_penalty = torch.mean(spline_penalty)
        
        return loss + \
            coef_entropy_penalty * entropy_penalty + \
                coef_indi_penalty * indi_penalty + \
                    coef_spline_penalty * spline_penalty
    
    @staticmethod
    def mixture_penalties(coef):
        # K: number of basis
        K = coef.shape[-1]
        device = coef.device
        # entropy penalty
        H = torch.sum(coef * torch.log(coef), axis = -1)  # batch X z_dim
        H0 = torch.tensor([0.95] + [0.05/(K-1)] * (K-1), device = device)
        H0 = torch.sum(H0 * torch.log(H0))
        entropy_penalty = F.relu(H - H0)
        # individual penalty
        alpha0 = torch.log(torch.tensor(0.01, device = device)) 
        indi_penalty = torch.sum(F.relu(alpha0 - torch.log(coef)), -1) # batch X z_dim
        return entropy_penalty.sum(-1), indi_penalty.sum(-1)
    
    @staticmethod
    def simpson(func, a, b, n=100):
        # numerical integral using Simpson's rule
        # assume a < b and n is an even positive integer
        h = (b-a)/n
        x = np.linspace(start=a, stop=b, num=n+1)
        
        if n==2:
            s = func(x[0]) + 4*func(x[1]) +func(x[2])
        else:
            s = func(x[0]) + func(x[n]) + 2*sum(func(x[1::2])) + \
                4*sum(func(x[2::2]))
        
        s = s*h/3
        return s

    @staticmethod
    def bspline_mcmc(bspl: BSpline, hops=300, n_burn_in = 0.5):
        states = []
        burn_in = int(hops*n_burn_in)

        lower_idx = np.nonzero(bspl.c)[0][0]
        init_samples = np.random.uniform(bspl.t[lower_idx], 
                                 bspl.t[lower_idx + bspl.k + 1],
                                 hops)
        current = init_samples[0]
        uni = np.random.uniform(0, 1, hops)

        for i in range(1, hops):
            states.append(current)
            movement = init_samples[i]

            curr_prob = bspl(x=current)
            move_prob = bspl(x=movement)

            acceptance = min(move_prob/curr_prob,1)
            if uni[i] <= acceptance:
                current = movement
        return states[burn_in:]
    
    def train_model(self, train_loader, optimizer, epoch, 
              T = 1, coef_entropy_penalty = 10, coef_spline_penalty=0.1 ,beta = 0.05,
                   temperature = 0.1):
        self.train()
        
        torch.manual_seed(epoch)
        
        train_loss = 0
        tloader = tqdm.tqdm(train_loader)
        
        for batch_idx, (data, _) in enumerate(tloader):
            
            if self.mnist_transform:
                data = data.to(self.device).view(-1, 784)
            
            optimizer.zero_grad()

            loss = self.calc_loss(data, T = T, 
                                  coef_entropy_penalty = coef_entropy_penalty, 
                                  coef_spline_penalty = coef_spline_penalty, 
                                  coef_indi_penalty = 0, beta = beta,
                                  temperature = temperature)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                tloader.set_postfix({"Epoch": epoch, "Loss": loss.item() / len(data)})
                        
        return train_loss / len(train_loader.dataset) # Average loss

    def test_model(self, test_loader, T = 1):
        self.eval()
        test_loss= 0
        with torch.no_grad():
            for data, _ in tqdm.tqdm(test_loader):
                if self.mnist_transform:
                    data = data.to(self.device).view(-1, 784)
                
                test_loss += self.calc_loss(data, T = T, beta = 0.0, 
                                            coef_entropy_penalty = 0, 
                                            coef_indi_penalty = 0).item()
            
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss
