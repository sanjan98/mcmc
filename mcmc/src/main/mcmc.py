"""
mcmc - Python code performing Markov Charin Monte Carlo and it's variants (delayed, adaptive and dram)

"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
from numpy.linalg.linalg import LinAlgError

### MH Acceptance ratio

def mh_acceptance_prob(current_target_logpdf,proposed_target_logpdf, current_sample, proposed_sample, proposal_func, cov):
    """Compute the metropolis-hastings accept-reject probability
    
    Inputs
    ------
    current_target_logpdf : float, logpdf at the current sample in the chain f_X(x^{(k)})
    proposed_target_logpdf : float, logpdf at the proposed sample in the chain
    current_sample : (d, ), current sample
    proposed_sample : (d, ), proposed sample
    proposal_func: f(x, y) callable that gives the log probability of y given x
    cov: (d, d), covariance of the proposal distribution
    
    Returns
    -------
    acceptance probability
    """
    prop_reverse = proposal_func(current_sample, proposed_sample, cov)
    prop_forward = proposal_func(proposed_sample, current_sample, cov)
    check = proposed_target_logpdf - current_target_logpdf + prop_reverse - prop_forward
    if check < 0:
        return np.exp(check)
    else:
        return 1      
    
### MH Delayed Acceptance ratio
    
def mh_acceptance_prob_delay(current_target_logpdf, proposed_target_logpdf, proposed_target2_logpdf, 
                             current_sample, proposed_sample, proposed_sample2, proposal_func, cov):
    """Compute the metropolis-hastings accept-reject probability
    
    Inputs
    ------
    current_target_logpdf : float, logpdf at the current sample in the chain f_X(x^{(k)})
    proposed_target_logpdf : float, logpdf at the proposed sample in the chain
    proposed_target2_logpdf : float, logpdf at the second (delayed) proposed sample in the chain
    current_sample : (d, ), current sample
    proposed_sample : (d, ), proposed sample
    proposed_sample2 : (d, ), second (delayed) proposed sample
    proposal_func: f(x, y) callable that gives the log probability of y given x
    cov: (d, d), covariance of the proposal distribution

    Returns
    -------
    acceptance probability
    """
    
    prop_reverse = proposal_func(proposed_sample, proposed_sample2, cov)
    prop_forward = proposal_func(proposed_sample, current_sample, cov)

    a1 = mh_acceptance_prob(proposed_target2_logpdf, proposed_target_logpdf, proposed_sample2, proposed_sample, proposal_func, cov)
    a2 = mh_acceptance_prob(current_target_logpdf, proposed_target_logpdf, current_sample, proposed_sample, proposal_func, cov)

    if a1 == 1:
        return 0
    if a2 == 1:
        return 1
    
    check = proposed_target2_logpdf - current_target_logpdf + prop_reverse - prop_forward + np.log(1-a1) - np.log(1-a2)
    if check < 0:
        return np.exp(check)
    else:
        return 1      

### DRAM Algo
    
def dram(starting_sample, proposal_cov, num_samples, target_logpdf, proposal_logpdf, proposal_sampler, adaptive=True, delayed =True,
         k0=100, gamma=0.5):
    """Delayed Adaaptive Metropolis-Hastings MCMC
    
    Inputs
    ------
    starting_sample: (d, ) the initial sample
    proposal_cov: (d, d) proposal covariance
    num_sample: positive integer, the number of total samples
    target_logpdf: function(x) -> logpdf of the target distribution
    proposal_logpdf: function (x, y) -> logpdf of proposing y if current sample is x
    proposal_sampler: function (x) -> y, generate a sample if you are currently at x
    adaptive, delayed: (logic) Switches for the Adaptive and Delayed Algo
    k0: Number of steps to wait until adaptation kicks in
    gamma: Hyper paramater multiplying the covariance in the delayed algo
    
    Returns
    -------
    Samples: (num_samples, d) array of samples
    accept_ratio: ratio of proposed samples that were accepted
    """

    d = starting_sample.shape[0]
    samples = np.zeros((num_samples, d))
    samples[0, :] = starting_sample
    current_target_logpdf = target_logpdf(samples[0, :])
    current_mean = starting_sample
    current_cov = proposal_cov
    
    num_accept = 0

    eps = 1e-7
    sd = (2.4**2/d)

    for ii in range(1, num_samples):
        # Check covariance condition
        if not is_positive_definite(current_cov):
            print(f'Caught non-positive definite matrix: {current_cov}')
            current_cov = nearest_positive_definite(current_cov)
        
        # propose
        proposed_sample = proposal_sampler(samples[ii-1, :], current_cov)
        proposed_target_logpdf = target_logpdf(proposed_sample)
        
        # determine acceptance probability
        a = mh_acceptance_prob(current_target_logpdf, proposed_target_logpdf, samples[ii-1,:], proposed_sample, proposal_logpdf, current_cov)
        
        # Accept or reject the sample
        u = np.random.rand()
        if a == 1 or u < a: #Accept
            samples[ii, :] = proposed_sample
            current_target_logpdf = proposed_target_logpdf
            num_accept += 1
        else:
            # Check if delayed is true
            if delayed:
                # second level proposal
                delayed_cov = current_cov*gamma
                proposed_sample2 = proposal_sampler(samples[ii-1, :], delayed_cov)
                proposed_target2_logpdf = target_logpdf(proposed_sample2)

                # Accept or reject the second sample based on delayed acceptance probability
                a_delay = mh_acceptance_prob_delay(current_target_logpdf, proposed_target_logpdf, proposed_target2_logpdf, 
                            samples[ii-1,:], proposed_sample, proposed_sample2, proposal_logpdf, delayed_cov)
                
                # Accept or reject the sample based on delayed acceptance prob
                u = np.random.rand()
                if a_delay == 1 or u < a_delay: #accept
                    samples[ii, :] = proposed_sample2
                    current_target_logpdf = proposed_target2_logpdf
                    num_accept += 1
                else: # reject
                    samples[ii, :] = samples[ii-1, :]
            
            else: # if delayed is false reject
                samples[ii, :] = samples[ii-1, :]
        
        # Check if adaptive is true
        if adaptive:
            # Update the sample mean every iteration!
            xbar_minus = current_mean.copy()
            current_mean = (1/(ii+1)) * samples[ii-1, :] + (ii/(ii+1))*xbar_minus

            # Start adapting covariance after k0 steps
            if ii >= k0:
                k = ii
                x = samples[ii, :]
                xk = x[:, np.newaxis]
                xbark = current_mean[:, np.newaxis]
                xbark_minus = xbar_minus[:, np.newaxis]
                current_cov = (k-1)/k * current_cov + sd/k * \
                              (k * np.multiply(xbar_minus.T, xbark_minus) - 
                               (k+1) * np.multiply(xbark.T, xbark) + np.multiply(xk.T, xk) + 
                               eps * np.eye(d))
           
    return samples, num_accept / float(num_samples-1)

## Additional Functions

def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except LinAlgError:
        return False
    
def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3
    
