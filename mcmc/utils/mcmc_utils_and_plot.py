"""Markov Chain Monte Carlo Plotting and Utilities

Copyright (c) Alex Gorodetsky, 2020
License: MIT
"""
import numpy as np
import functools
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import scipy
from scipy import stats

import matplotlib as mpl
from matplotlib import cm
from collections import OrderedDict
from collections import namedtuple
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# For comments on the first four functions see the Gaussian Random Variable Notebook
def lognormpdf(x, mean, cov):
    """Compute the log pdf of a Normal distribution
    
    Inputs
    ------
    x : (d, N) variable of interest
    mean : (d, ) mean of the distribution
    cov  : (d, d) covariance of the distribution
    
    Returns
    -------
    logpdf: (float) log pdf value
    """

    if len(x.shape)==1:
        x = x[:, np.newaxis] # This is required to change from tuple to array
    d, N = x.shape
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.linalg.det(cov)**0.5
    diff = x - np.tile(mean[:, np.newaxis], (1, N))
    sol = np.linalg.solve(cov, diff)
    inexp = np.einsum("ij,ij->j",diff, sol)
    logpdf = np.log(preexp) - 0.5 * inexp
    return logpdf

def build_cov_mat(std1, std2, rho):
    """Build a covariance matrix for a bivariate Gaussian distribution
    
    Inputs
    ------
    std1 : positive real, standard deviation of first variable
    std2 : positive real, standard deviation of second variable
    rho  : real number between [-1, 1] representing the correlation
    
    Returns
    -------
    Bivariate covariance Matrix
    """
    assert std1 > 0, "standard deviation must be greater than 0"
    assert std2 > 0, "standard deviation must be greater than 0"
    assert np.abs(rho) <= 1, "correlation must be betwene -1 and 1"
    return np.array([[std1**2, rho * std1 * std2], [rho * std1 * std2, std2**2]])

def eval_normpdf_on_grid(x, y, mean, cov):
    XX, YY = np.meshgrid(x,y)
    pts = np.stack((XX.reshape(-1), YY.reshape(-1)),axis=0)
    evals = np.exp(lognormpdf(pts, mean, cov).reshape(XX.shape))
    return XX, YY, evals

def eval_func_on_grid(func, gridx, gridy):
    "Evaluate the function *func* on a grid discretized by gridx and gridy"
    vals = np.zeros((gridx.shape[0], gridy.shape[0]))
    for ii in range(gridx.shape[0]):
        for jj in range(gridy.shape[0]):
            pt = np.array([gridx[ii], gridy[jj]])
            vals[ii, jj] = func(pt)
            
    return vals   

def plot_bivariate_gauss(x, y, mean, cov, axis=None):
    std1 = cov[0,0]**0.5
    std2 = cov[1,1]**0.5
    mean1 = mean[0]
    mean2 = mean[1]
    XX, YY, evals = eval_normpdf_on_grid(x, y, mean, cov)
    if axis is None:
        fig, axis = plt.subplots(2,2, figsize=(10,10))
        
    axis[0,0].plot(x, np.exp(lognormpdf(x[np.newaxis,:], np.array([mean1]), np.array([[std1**2]]))))
    axis[0,0].set_ylabel(r'$f_{X_1}$')
    axis[1,1].plot(np.exp(lognormpdf(y[np.newaxis,:], np.array([mean2]), np.array([[std2**2]]))),y)
    axis[1,1].set_xlabel(r'$f_{X_2}$')
    axis[1,0].contourf(XX, YY, evals)
    axis[1,0].set_xlabel(r'$x_1$')
    axis[1,0].set_ylabel(r'$x_2$')
    axis[0,1].set_visible(False)
    return fig, axis

def sub_sample_data(samples, frac_burn=0.2, frac_use=0.7):
    """Subsample data by burning off the front fraction and using another fraction

    Inputs
    ------
    samples: (N, d) array of samples
    frac_burn: fraction < 1, percentage of samples from the front to ignore
    frac_use: percentage of samples to use after burning, uniformly spaced
    """
    nsamples = samples.shape[0]
    inds = np.arange(nsamples, dtype=np.int)
    start = int(frac_burn * nsamples)
    inds = inds[start:]
    nsamples = nsamples - start
    step = int(nsamples / (nsamples * frac_use))
    inds2 = np.arange(0, nsamples, step)
    inds = inds[inds2]
    return samples[inds, :]

def scatter_matrix(samples, #list of chains
                   mins=None, maxs=None,
                   upper_right=None,
                   specials=None,
                   hist_plot=True, # if false then only data
                   nbins=200,
                   gamma=0.5,
                   labels=None):

    nchains = len(samples)
    dim = samples[0].shape[1]
    

    if mins is None:
        mins = np.zeros((dim))
        maxs = np.zeros((dim))

        for ii in range(dim):
            # print("ii = ", ii)
            mm = [np.quantile(samp[:, ii], 0.01, axis=0) for samp in samples]
            # print("\t mins = ", mm)
            mins[ii] = np.min(mm)
            mm = [np.quantile(samp[:, ii], 0.99, axis=0) for samp in samples]            
            # print("\t maxs = ", mm)
            maxs[ii] = np.max(mm)

            if specials is not None:
                if isinstance(specials, list):
                    minspec = np.min([spec['vals'][ii] for spec in specials])
                    maxspec = np.max([spec['vals'][ii] for spec in specials])
                else:
                    minspec = spec['vals'][ii]
                    maxspec = spec['vals'][ii]
                mins[ii] = min(mins[ii], minspec)
                maxs[ii] = max(maxs[ii], maxspec)
    

    deltas = (maxs - mins) / 10.0
    use_mins = mins - deltas
    use_maxs = maxs + deltas

    cmuse = cm.get_cmap(name='tab10')

    # fig = plt.figure(constrained_layout=True)
    fig = plt.figure(figsize=(18,10))
    if upper_right is None:
        gs = GridSpec(dim, dim, figure=fig)
        axs = [None]*dim*dim
        start = 0
        end = dim
        l = dim
    else:
        gs = GridSpec(dim+1, dim+1, figure=fig)
        axs = [None]*(dim+1)*(dim+1)
        start = 1
        end = dim + 1
        l = dim+1

    means = [np.mean(np.concatenate([samples[kk][:, ii] for kk in range(nchains)])) for ii in range(dim)]

    def one_decimal(x, pos):
        return f'{x:.1f}'

    formatter = FuncFormatter(one_decimal)

    # print("mins = ", mins)
    # print("maxs = ", maxs)
    for ii in range(dim):
        # print("ii = ", ii)
        axs[ii] = fig.add_subplot(gs[ii+start, ii])
        ax = axs[ii]

        # Turn everythinng off
        if ii < dim-1:
            ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
            if labels:
                ax.set_xlabel(labels[ii], fontsize='14')
            
        ax.tick_params(axis='y', left=False, right=False, labelleft=False)
        ax.set_frame_on(False)

        sampii = np.concatenate([samples[kk][:, ii] for kk in range(nchains)])
        # for kk in range(nchains):
        # print("sampii == ", sampii)
        ax.hist(sampii,            
                # ax.hist(samples[kk][:, ii],
                bins='sturges',
                density=True,
                edgecolor='black',
                stacked=True,
                range=(use_mins[ii],use_maxs[ii]),
                alpha=0.4)
        if specials is not None:
            for special in specials:
                if special['vals'][ii] is not None:
                    # ax.axvline(special[ii], color='red', lw=2)
                    if 'color' in special:
                        ax.axvline(special['vals'][ii], color=special['color'], lw=2)
                    else:
                        ax.axvline(special['vals'][ii], lw=2)
        
        ax.axvline(means[ii], color='red', linestyle='--', lw=2, label=f'Mean: {means[ii]:.2f}')
        ax.set_xlim((use_mins[ii]-1e-10, use_maxs[ii]+1e-10))

        # Setting two tick marks manually
        diff = 0.2*(use_maxs[ii]-use_mins[ii])
        xticks = np.linspace(use_mins[ii]+diff, use_maxs[ii]-diff, 2)
        yticks = ax.get_yticks()
        if len(yticks) >= 2:
            yticks = np.linspace(yticks[0], yticks[-1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # Setting the formatter
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        for jj in range(ii+1, dim):
            # print("jj = ", jj)
            axs[jj*l + ii] = fig.add_subplot(gs[jj+start, ii])
            ax = axs[jj*l + ii]


            if jj < dim-1:
                ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
                if labels:
                    ax.set_xlabel(labels[ii], fontsize='14')
            if ii > 0:
                ax.tick_params(axis='y', left=False, right=False, labelleft=False)
            else:
                ax.tick_params(axis='y', left=True, right=False, labelleft=True)
                if labels:
                    ax.set_ylabel(labels[jj], fontsize='14')
                    
            ax.set_frame_on(True)     

            for kk in range(nchains):
                if hist_plot is True:
                    ax.hist2d(samples[kk][:, ii], samples[kk][:, jj],
                              bins=nbins,
                              norm=mcolors.PowerNorm(gamma),
                              density=True,
                              cmap=plt.cm.jet)
                    # ax.tick_params(axis='x', which='major', labelsize=6)
                else:
                    ax.plot(samples[kk][:, ii], samples[kk][:, jj], 'o', ms=1, alpha=gamma)

                # ax.hist2d(samples[kk][:, ii], samples[kk][:, jj], bins=nbins)

            if specials is not None:
                for special in specials:
                    if 'color' in special:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x',
                                color=special['color'], ms=2, mew=2)
                    else:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x',
                                ms=2, mew=2)
            
            # ax.axhline(means[jj], color='red', linestyle='--', lw=2)
            # ax.axvline(means[ii], color='red', linestyle='--', lw=2)

            ax.set_xlim((use_mins[ii], use_maxs[ii]))
            ax.set_ylim((use_mins[jj]-1e-10, use_maxs[jj]+1e-10))

            # Setting two tick marks manually
            diff = 0.2*(use_maxs[ii]-use_mins[ii])
            xticks = np.linspace(use_mins[ii]+diff, use_maxs[ii]-diff, 2)
            yticks = np.linspace(use_mins[jj]+diff, use_maxs[jj]-diff, 2)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            # Setting the formatter
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout(pad=0.01);
    if upper_right is not None:
        size_ur = int(dim/2)

        name = upper_right['name']
        vals = upper_right['vals']
        if 'log_transform' in upper_right:
            log_transform = upper_right['log_transform']
        else:
            log_transform = None
        ax = fig.add_subplot(gs[0:int(dim/2),
                                size_ur+1:size_ur+int(dim/2)+1])

        lb = np.min([np.quantile(val, 0.01) for val in vals])
        ub = np.max([np.quantile(val, 0.99) for val in vals])
        for kk in range(nchains):
            if log_transform is not None:
                pv = np.log10(vals[kk]) 
                ra = (np.log10(lb), np.log10(ub))
            else:
                pv = vals[kk]
                ra = (lb, ub)
            ax.hist(pv,
                    density=True,
                    range=ra,
                    edgecolor='black',
                    stacked=True,
                    bins='auto',
                    alpha=0.2)
        ax.tick_params(axis='x', bottom='both', top=False, labelbottom=True)
        ax.tick_params(axis='y', left='both', right=False, labelleft=False)
        ax.set_frame_on(True)
        ax.set_xlabel(name, fontsize='14')

        # Setting two tick marks manually for the upper-right subplot
        diff = 0.2*(ra[1]-ra[0])
        xticks = np.linspace(ra[0]+diff, ra[1]-diff, 2)
        yticks = ax.get_yticks()
        if len(yticks) >= 2:
            yticks = np.linspace(yticks[0], yticks[-1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # Setting the formatter
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)


    plt.subplots_adjust(left=0.15, right=0.95)
    return fig, axs, gs

def plot_trace(samples, labels=None, output_fname='.' ):
    dim = samples.shape[1]
    fig, axs = plt.subplots(dim, 1, figsize=(10,5))
    for i in range(dim):
        axs[i].plot(samples[:, i], '-k')
        axs[i].set_ylabel(f'{labels[i]}', fontsize=14)
        if i < dim-1:
            axs[i].set_xticks([])    
    axs[dim-1].set_xlabel('Sample Number', fontsize=14)
    plt.savefig(f'{output_fname}.png', bbox_inches='tight')

def plot_lag(samples, lbl=None, output_fname='.', maxlag=500, step=1):
    lags, autolag = autocorrelation(samples, maxlag=maxlag,step=step)
    ess = []
    dim = samples.shape[1]
    fig, axs = plt.subplots(dim, 1, figsize=(10,5))
    for i in range(dim):
        ess.append(effective_sample_size(autolag[:,i]))
        axs[i].plot(lags, autolag[:, i],'-o', label='ESS = %.2f' % ess[i], alpha=0.3)
        axs[i].set_ylabel(f'{lbl[i]}', fontsize=14)
        axs[i].legend(fontsize=11)
        if i < dim-1:
            axs[i].set_xticks([])
    axs[dim-1].set_xlabel('Lag', fontsize=14)
    plt.savefig(f'{output_fname}.png', bbox_inches='tight')

### Auto Correlation

def autocorrelation(samples, maxlag=100, step=1):
    """Compute the correlation of a set of samples
    
    Inputs
    ------
    samples: (N, d)
    maxlag: maximum distance to compute the correlation for
    step: step between distances from 0 to maxlag for which to compute teh correlations

    Returns
    -------
    lags: evenly space lag from 0 to maxlag
    autos: Auto-correlation R(lags)
    """
    
    # Get the shapes
    try:
        ndim = samples.shape[1]
    except IndexError:
        ndim = 1
        samples = samples[:, np.newaxis]
    nsamples = samples.shape[0]    
    
    # Compute the mean
    mean = np.mean(samples, axis=0)
    
    # Compute the denominator, which is variance
    denominator = np.zeros((ndim))
    for ii in range(nsamples):
        denominator = denominator + (samples[ii, :] - mean)**2
    
    lags = np.arange(0, maxlag, step)
    autos = np.zeros((len(lags), ndim))
    for zz, lag in enumerate(lags):
        autos[zz, :] = np.zeros((ndim))
        # compute the covariance between all samples *lag apart*
        for ii in range(nsamples - lag):
            autos[zz,:] = autos[zz, :] + (samples[ii,:]-mean)*(samples[ii + lag,:] -mean)
        autos[zz, :] = autos[zz, :]/denominator
    return lags, autos


### Effective Sample Size

def effective_sample_size(auto_corrs):
    """Estimate the effective sample size for an array of samples."""
    n = len(auto_corrs)

    # Sum the sequence of autocorrelations
    negative_autocorr = auto_corrs[auto_corrs < 0]
    if len(negative_autocorr) > 0:  # truncate the sum at first negative autocorrelation
        first_negative = np.where(auto_corrs < 0)[0][0]
    else:
        first_negative = len(auto_corrs)

    ess = n / (1 + 2 * np.sum(auto_corrs[:first_negative]))
    return ess

def batch_normal_pdf(x, mu, cov, logpdf=True):
    """
    Compute the multivariate normal pdf at each x location.
    Dimensions
    ----------
    d: dimension of the problem
    *: any arbitrary shape (a1, a2, ...)
    Parameters
    ----------
    x: (*, d) location to compute the multivariate normal pdf
    mu: (*, d) mean values to use at each x location 
    cov: (*, d, d) covariance matrix
    Returns
    -------
    pdf: (*) the multivariate normal pdf at each x location
    """
    # Make some checks on input
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    cov = np.atleast_1d(cov)
    dim = cov.shape[-1]

    # 1-D case
    if len(cov.shape) == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    if len(mu.shape) == 1:
        mu = mu[np.newaxis, :]

    assert cov.shape[-1] == cov.shape[-2] == dim
    assert x.shape[-1] == mu.shape[-1] == dim

    # Normalizing constant (scalar)
    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))

    # Can broadcast x - mu with x: (1, Nr, Nx, d) and mu: (Ns, Nr, Nx, d)
    diff = x - mu

    # In exponential
    diff_col = diff.reshape((*diff.shape, 1))  # (*, d, 1)
    diff_row = diff.reshape((*diff.shape[:-1], 1, diff.shape[-1]))  # (*, 1, d)
    inexp = np.squeeze(diff_row @ np.linalg.inv(cov) @ diff_col, axis=(-1, -2))  # (*, 1, d) x (*, d, 1) = (*, 1, 1)

    # Compute pdf
    pdf = np.log(preexp) + (-1/2)*inexp if logpdf else preexp * np.exp(-1 / 2 * inexp)

    return pdf.astype(np.float32)


def batch_normal_sample(mean, cov, size: "tuple | int" = ()):
    """
    Batch sample multivariate normal distributions.
    https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i
    Arguments:
        mean: expected values of shape (…M, D)
        cov: covariance matrices of shape (…M, D, D)
        size: additional batch shape (…B)
    Returns: samples from the multivariate normal distributions
             shape: (…B, …M, D)
    """
    # Make some checks on input
    mean = np.atleast_1d(mean)
    cov = np.atleast_1d(cov)
    dim = cov.shape[0]

    # 1-D case
    if dim == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(mean.shape) == 1:
        mean = mean[np.newaxis, :]

    assert cov.shape[0] == cov.shape[1] == dim
    assert mean.shape[-1] == dim

    size = (size, ) if isinstance(size, int) else tuple(size)
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    X = np.random.standard_normal((*shape, 1)).astype(np.float32)
    L = np.linalg.cholesky(cov)
    sample = (L @ X).reshape(shape) + mean
    if dim == 1:
        sample = np.squeeze(sample, axis=-1)
    return sample

def normal_sample(mean, cov, nsamples=1):
    """Generate nsamples from a multivariate Normal distribution

    Inputs
    ------
    mean: (d, ) Mean of distribution
    cov: (d, d) Covariance of distribution
    nsamples: (int) Number of samples
    
    Returns
    -------
    samples: (d, nsamples) Column vector of samples
    """

    # Generate standard normal samples
    dim = len(mean)
    standard_normal_samples = np.random.randn(dim, nsamples)
    # Apply Cholesky factorization on the covariance matrix
    try:
        cholesky_factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = nearest_positive_definite(cov)
        cholesky_factor = np.linalg.cholesky(cov)
    # Generate the samples
    samples = mean[:,np.newaxis] + cholesky_factor @ standard_normal_samples
    samples = np.squeeze(samples)
    return samples

def compose(*functions):
    "Compose a list of functions"
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def banana_logpdf(x,a=1.0,b=100.0):
    logpdf = (a-x[0])**2 + b * (x[1] - x[0]**2)**2
    #logpdf = np.exp(-0.5 * (x[0])**2 + x[1]**2) - np.exp(-0.5 / 1.0 * (x[0]**2 + x[1]**2))
    #logpdf = np.log(logpdf)
    #logpdf = (np.sin(10*x[0]*x[1]) + x[1]**2)*4
    return -logpdf
    #return logpdf

def plot_banana():
    plt.figure()
    xgrid = np.linspace(-1, 2, 100)
    ygrid = np.linspace(-1, 2, 100)
    XX, YY = np.meshgrid(xgrid, ygrid)
    plt.contourf(XX, YY, 
                 eval_func_on_grid(compose(np.exp, banana_logpdf), 
                                   xgrid, ygrid).T)
    
def laplace_approx(x0, logpost, optmethod):
    """Perform the laplace approximation, returning the MAP point and an approximation of the covariance
    :param x0: (nparam, ) array of initial parameters
    :param logpost: f(param) -> log posterior pdf

    :returns map_point: (nparam, ) MAP of the posterior
    :returns cov_approx: (nparam, nparam), covariance matrix for Gaussian fit at MAP
    """
    # Gradient free method to obtain optimum
    neg_post = lambda x: -logpost(x)
    # neg_post = lambda x: -logpost(x)
    res = scipy.optimize.minimize(neg_post, x0, method=optmethod)#, tol=1e-6, options={'maxiter': 100, 'disp': True})
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print('--------------------First optimization done---------------------------------')
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    # Gradient method which also approximates the inverse of the hessian
    # res = scipy.optimize.minimize(neg_post, res.x*0.95, method=optmethod)#, tol=1e-4, options={'maxiter': 20, 'disp': True})
    map_point = res.x
    cov_approx = res.hess_inv
    return map_point, cov_approx

def log_banana(x,co):
    if (len(x.shape) == 1):
        x = x[np.newaxis, :]
    N, d = x.shape
    x1p = x[:, 0]
    x2p = x[:, 1] + (np.square(x[:, 0]) + 1)
    xp = np.concatenate((x1p[:, np.newaxis], x2p[:, np.newaxis]), axis=1)
    sigma = np.array([[1, 0.9], [0.9, 1]])
    mu = np.array([0, 0])
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.linalg.det(sigma)**0.5
    diff = xp - np.tile(mu[np.newaxis, :], (N, 1))
    sol = np.linalg.solve(sigma, diff.T)
    inexp = np.einsum("ij,ij->j", diff.T, sol)
    co+=1
    return np.log(preexp) - 0.5 * inexp, co

def lognormpdf_univariate(x, mean, cov):
    """Compute the log pdf of a univariate Normal distribution
    
    Inputs
    ------
    x : (float) variable of interest
    mean : (float) mean of the distribution
    cov  : (float) covariance of the distribution
    
    Returns
    -------
    logpdf: (float) log pdf value
    """

    preexp = 1.0 / (2.0 * np.pi)**(0.5) / (cov)**0.5
    diff = x - mean
    inexp = diff**2/cov
    logpdf = np.log(preexp) - 0.5 * inexp
    return logpdf

def invgamma_univariate(x, alpha, beta):
    """Compute the log pdf of a univariate Inverse Gamma distribution
    
    Inputs
    ------
    x : (float) variable of interest
    alpha : (float) shape parameter
    beta  : (float) scale parameter
    
    Returns
    -------
    logpdf: (float) log pdf value
    """

    logpdf = alpha*np.log(beta) - scipy.special.gammaln(alpha) - (alpha+1)*np.log(x) - beta/x
    return logpdf

def nearest_positive_definite(A):
    """Find the nearest positive definite matrix to A."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T * s, V)

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def is_positive_definite(A):
    """Check if a matrix A is positive definite by attempting Cholesky decomposition."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
def batched_variance(data, batch_size):
    """Calculates the variance of a dataset using a batched approach."""

    n_batches = int(np.ceil(len(data) / batch_size))
    batch_variances = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(data))
        batch = data[start:end]
        batch_variances.append(np.var(batch, ddof=1))  # Sample variance

    return np.mean(batch_variances)