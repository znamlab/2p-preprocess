import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.special as sp
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam

import matplotlib.pyplot as plt


# taken from example files of autograd package
def black_box_variational_inference(logprob, n_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    rs = npr.RandomState(0)

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        # Variational dist is a diagonal Gaussian.
        D = params.size // 2
        mean, log_std = params[:D], params[D:]
        samples = rs.randn(n_samples, D) * np.exp(log_std) + mean
        gaussian_entropy = 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)
        lower_bound = gaussian_entropy + np.mean(logprob(samples, t))
        return -lower_bound

    return variational_objective


def K(nu):
    """helper function for asymmetric Student pdf"""
    return (
        sp.gamma(0.5 * (nu + 1)) / (np.sqrt(np.pi * nu) * sp.gamma(0.5 * nu))
    )


# from Zhu and Galbraith - 2009 - A Generalized Asymmetric Student-t
# Distribution with Application to Financial Econometrics
def log_density_ast(y, alpha, nu1, nu2, mu, sigma):
    """log pdf of asymmetric Student distribution"""
    z = (y - mu) / (2 * sigma)
    left_branch = (
        -np.log(sigma)
        - 0.5 * (nu1 + 1) * np.log1p((z / (alpha * K(nu1)))**2 / nu1)
    )
    right_branch = (
        -np.log(sigma)
        - 0.5 * (nu2 + 1) * np.log1p((z / ((1 - alpha) * K(nu2)))**2 / nu2)
    )
    return np.where(z < 0, left_branch, right_branch)


def ast_model(traces, n_sectors, n_samples=1, n_iters=5000, lr=0.01,
              verbose=False):
    """asymmetric Student model inferred with black-box SVI"""

    nsig, nt = traces.shape
    scale = traces.std()
    traces = traces / scale

    def unpack_x(x):
        log_alpha = x[..., :1, np.newaxis]
        offset = x[..., 1:1+nsig, np.newaxis]
        mu = x[..., np.newaxis, 1+nsig:-1]
        log_sigma = x[..., -1:, np.newaxis]
        return log_alpha, offset, mu, log_sigma

    def transform_x(log_alpha, offset, mu, log_sigma):
        alpha = norm.cdf(log_alpha)
        sigma = np.exp(log_sigma)
        return alpha, offset, mu, sigma

    def log_density(x, t):
        log_alpha, offset, mu, log_sigma = unpack_x(x)
        alpha, offset, mu, sigma = transform_x(
            log_alpha, offset, mu, log_sigma
        )
        alpha2 = np.concatenate([alpha, np.ones_like(alpha)], 1)

        # prior
        alpha_density = norm.logpdf(log_alpha, 0, 1)
        mu_density = norm.logpdf(mu, 0, 1)
        offset_density = norm.logpdf(offset, 0, 1)
        sigma_density = norm.logpdf(log_sigma, 0, 1)

        # likelihood
        sigmas = sigma / np.sqrt(n_sectors[:, np.newaxis])
        x_density = log_density_ast(
            traces, 0.5, 30, 1, alpha2 * mu + offset, sigmas
        )

        return (
            sigma_density.squeeze() + alpha_density.squeeze() +
            mu_density.squeeze().sum(axis=-1) +
            offset_density.squeeze().sum(axis=-1) +
            x_density.sum(axis=-1).sum(axis=-1)
        )

    # variational objective
    objective = black_box_variational_inference(log_density, n_samples)
    gradient = grad(objective)

    # initial variational parameters
    D = nsig * 2 + nt
    init_mean = -1 * np.ones(D)
    init_log_std = -5 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])

    # create callback function
    callback = None

    if verbose is True:
        def callback(params, t, g):
            elbo = -objective(params, t)
            print("Iteration {} lower bound {}".format(t, elbo))

    elif verbose == 'display':

        class Callback:

            def __init__(self):
                self.elbos = []
                _, self.axes = plt.subplots(4, 1, figsize=(12, 10))

            def __call__(self, params, t, g):
                elbo = -objective(params, t)
                self.elbos.append(elbo)

                print("Iteration {} lower bound {}".format(t, elbo))
                if t % 10 == 0:
                    self.axes[0].clear()
                    self.axes[0].plot(self.elbos)

                    alpha, offset, mu, _ = transform_x(*unpack_x(params[:D]))
                    self.axes[1].clear()
                    self.axes[1].plot(mu[0])
                    self.axes[2].clear()
                    self.axes[2].plot(traces[0])
                    self.axes[2].plot(traces[0] - alpha[0] * mu[0])
                    self.axes[3].clear()
                    self.axes[3].plot(traces[1])
                    self.axes[3].plot(traces[1] - mu[0])

                plt.pause(1e-5)

        callback = Callback()

    # optimization of the variational objective
    var_params = adam(
        gradient, init_var_params, step_size=lr, num_iters=n_iters,
        callback=callback
    )

    # clean trace
    alpha, _, mu, _ = transform_x(*unpack_x(var_params[:D]))
    cleaned_trace = (traces[0] - alpha[0] * mu[0]) * scale

    return cleaned_trace, var_params
