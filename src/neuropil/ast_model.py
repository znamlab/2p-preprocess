import jax.numpy as jnp
import jax.random as npr
import scipy.special as sp
import jax.scipy.stats.norm as norm
from jax import value_and_grad, grad, jit
from jax.experimental.optimizers import adam


# taken from example files of autograd package
def black_box_variational_inference(logprob, n_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def variational_objective(params, t, traces):
        """Provides a stochastic estimate of the variational lower bound."""
        rng = npr.PRNGKey(t)
        # Variational dist is a diagonal Gaussian.
        D = params.size // 2
        mean, log_std = params[:D], params[D:]
        samples = npr.normal(rng, [n_samples, D]) * jnp.exp(log_std) + mean
        # only part of the entropy that depends on params
        gaussian_entropy = jnp.sum(log_std)
        lower_bound = gaussian_entropy + jnp.mean(logprob(samples, traces))
        return -lower_bound

    return variational_objective


def K(nu):
    """helper function for asymmetric Student pdf"""
    # this part does not get differentiated so using the scipy method
    # the scipy functions will be executed at compile time and their output
    # will be passed to the GPU
    return (
        sp.gamma(0.5 * (nu + 1)) / (jnp.sqrt(jnp.pi * nu) * sp.gamma(0.5 * nu))
    )


# from Zhu and Galbraith - 2009 - A Generalized Asymmetric Student-t
# Distribution with Application to Financial Econometrics
def log_density_ast(y, alpha, nu1, nu2, mu, sigma):
    """log pdf of asymmetric Student distribution"""
    z = (y - mu) / (2 * sigma)
    left_branch = (
        -jnp.log(sigma)
        - 0.5 * (nu1 + 1) * jnp.log1p((z / (alpha * K(nu1)))**2 / nu1)
    )
    right_branch = (
        -jnp.log(sigma)
        - 0.5 * (nu2 + 1) * jnp.log1p((z / ((1 - alpha) * K(nu2)))**2 / nu2)
    )
    return jnp.where(z < 0, left_branch, right_branch)


def ast_model(traces, n_sectors, n_samples=1, n_iters=5000, lr=0.01,
              verbose=False):
    """asymmetric Student model inferred with black-box SVI"""
    nsig, nt = traces.shape
    scale = traces.std()
    traces = traces / scale

    def unpack_x(x):
        log_alpha = x[..., :1, jnp.newaxis]
        offset = x[..., 1:1+nsig, jnp.newaxis]
        mu = x[..., jnp.newaxis, 1+nsig:-1]
        log_sigma = x[..., -1:, jnp.newaxis]
        return log_alpha, offset, mu, log_sigma

    def transform_x(log_alpha, offset, mu, log_sigma):
        alpha = norm.cdf(log_alpha)
        sigma = jnp.exp(log_sigma)
        return alpha, offset, mu, sigma

    def log_density(x, traces):
        log_alpha, offset, mu, log_sigma = unpack_x(x)
        alpha, offset, mu, sigma = transform_x(
            log_alpha, offset, mu, log_sigma
        )
        alpha2 = jnp.concatenate([alpha, jnp.ones_like(alpha)], 1)

        # prior
        alpha_density = norm.logpdf(log_alpha, 0, 1)
        mu_density = norm.logpdf(mu, 0, 1)
        offset_density = norm.logpdf(offset, 0, 1)
        sigma_density = norm.logpdf(log_sigma, 0, 1)

        # likelihood
        sigmas = sigma / jnp.sqrt(n_sectors[:, jnp.newaxis])
        # this is where we pass data to JAX - awkward...
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

    # initial variational parameters
    D = nsig * 2 + nt
    init_mean = -1 * jnp.ones(D)
    init_log_std = -5 * jnp.ones(D)
    init_var_params = jnp.concatenate([init_mean, init_log_std])

    # this decorator tells JAX to use XLA to compile the code
    @jit
    def update(params, opt_state, t):
        """ Compute the gradient and update the parameters """
        value, grads = value_and_grad(objective)(params, t, traces)
        opt_state = opt_update(t, grads, opt_state)
        return get_params(opt_state), opt_state, value

    # optimization of the variational objective
    opt_init, opt_update, get_params = adam(lr)
    opt_state = opt_init(init_var_params)
    params = get_params(opt_state)
    for i in range(n_iters):
        params, opt_state, loss = update(params, opt_state, i)
        if i+1 % 1000 == 0:
            print("Iteration {} lower bound {}".format(i+1, loss))

    # clean trace
    alpha, _, mu, _ = transform_x(*unpack_x(params[:D]))
    cleaned_trace = (traces[0] - alpha[0] * mu[0]) * scale

    return cleaned_trace, params
