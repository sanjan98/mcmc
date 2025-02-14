"""
mcmc - Python code performing Markov Charin Monte Carlo and it's variants (delayed, adaptive and dram)

"""
# Imports
import os
import numpy as np
from numpy.linalg.linalg import LinAlgError
from typing import Tuple

# Metropolis Hastings Acceptance ratio
def mh_acceptance_prob(current_logpdf: float, proposed_logpdf: float, current_sample: np.ndarray, proposed_sample: np.ndarray, proposal_pdf: callable, cov: np.ndarray) -> float:
    """Compute the metropolis-hastings accept-reject probability
    
    Inputs
    ------
    current_logpdf : float, logpdf of the current sample
    proposed_logpdf : float, logpdf of the proposed sample
    current_sample : (d, ), current sample
    proposed_sample : (d, ), proposed sample
    proposal_pdf: f(x, y) -> float callable that gives the log probability of y given x
    cov: (d, d), covariance of the proposal distribution
    
    Returns
    -------
    acceptance probability
    """
    prop_reverse = proposal_pdf(current_sample, proposed_sample, cov)
    prop_forward = proposal_pdf(proposed_sample, current_sample, cov)
    check = proposed_logpdf - current_logpdf + prop_reverse - prop_forward
    if check < 0:
        return np.exp(check)
    else:
        return 1    
    
### MH Delayed Acceptance ratio
    
def delay_acceptance_prob(current_logpdf: float, proposed_logpdf: float, proposed2_logpdf: float, current_sample: np.ndarray, proposed_sample: np.ndarray, proposed_sample2: np.ndarray, proposal_func: callable, cov: np.ndarray) -> float:
    """Compute the metropolis-hastings accept-reject probability
    
    Inputs
    ------
    current_logpdf : float, logpdf of the current sample
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
    
    # prop_reverse = proposal_func(proposed_sample, proposed_sample2, cov)
    # prop_forward = proposal_func(proposed_sample, current_sample, cov)
    
    a21 = mh_acceptance_prob(proposed2_logpdf, proposed_logpdf, proposed_sample2, proposed_sample, proposal_func, cov)
    if a21 > 1.0-1e-15: # reject because current is more likely
        return 0
    
    prop_pdf_num = proposal_func(proposed_sample2, proposed_sample, cov)
    prop_pdf_den = proposal_func(current_sample, proposed_sample, cov)

    a2 = proposed2_logpdf - current_logpdf + prop_pdf_num - prop_pdf_den + np.log(1.0-a21) - np.log(1.0 - min(1, np.exp(proposed_logpdf - current_logpdf)))

    if a2 < 0:
        return np.exp(a2)
    else:    
        return 1

    # a1 = mh_acceptance_prob(proposed2_logpdf, proposed_logpdf, proposed_sample2, proposed_sample, proposal_func, cov)
    # a2 = mh_acceptance_prob(current_logpdf, proposed_logpdf, current_sample, proposed_sample, proposal_func, cov)

    # if a1 == 1:
    #     return 0
    # if a2 == 1:
    #     return 1
    
    # check = proposed2_logpdf - current_logpdf + prop_reverse - prop_forward + np.log(1-a1) - np.log(1-a2)
    # if check < 0:
    #     return np.exp(check)
    # else:
    #     return 1      

### DRAM Algo
    
def dram(starting_sample: np.ndarray, cov: np.ndarray, num_samples: int, target_logpdf: callable, proposal_logpdf: callable, sampler: callable, output_fname: str, adaptive: bool = True, delayed: bool = True, k0: int = 100, gamma: float = 0.5, cost: int = 0, save_counter: int = 100) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """Delayed Rejection Adaptive Metropolis-Hastings MCMC (DRAM MCMC)
    
    Inputs
    ------
    starting_sample: (d, ) the initial sample
    cov: (d, d) proposal covariance
    num_samples: positive integer, the number of total samples
    target_logpdf: function(x) -> (float, int), logpdf of the target distribution and cost of the target logpdf
    proposal_logpdf: function (x, y) -> float, logpdf of proposing y if current sample is x
    proposal_sampler: function (x) -> y, generate a sample if you are currently at x
    output_fname: str, name of the file to save the samples and covariance
    adaptive, delayed: (logic) Switches for the Adaptive and Delayed Algo
    k0: Number of steps to wait until adaptation kicks in
    gamma: Hyper paramater multiplying the covariance in the delayed algo
    cost: int, sequential cost of the target logpdf
    save_counter: int, save the samples and covariance every save_counter iterations (for restart)
    
    Returns
    -------
    Samples: (num_samples, d) array of samples
    Covariance: (num_samples, d, d) history of covariance matrix
    accept_ratio: ratio of proposed samples that were accepted
    cost: int, cost of the whole algorithm
    """
    if isinstance(starting_sample, float)==True or isinstance(starting_sample, int)==True:
        raise ValueError("Starting sample should be an array not a float or int")
    else:
        # Get dimension of the problem
        dim = starting_sample.shape[0]

    samples = np.zeros((num_samples, dim))
    covar = np.zeros((num_samples, dim, dim))
    samples[0, :] = starting_sample
    
    current_logpdf, cost = target_logpdf(samples[0, :],cost)
    current_mean = starting_sample
    current_cov = cov
    covar[0, :, :] = current_cov
    
    num_accept = 0

    # Adaptation parameters (hardcoded for now)
    eps = 1e-7
    sd = (2.4**2/dim)

    for ii in range(1, num_samples):

        if np.mod(ii,100) == 0:
            print(f"-------------------In Iteration {ii}--------------")
        
        # propose
        proposed_sample = sampler(samples[ii-1, :], current_cov)
        proposed_logpdf, cost = target_logpdf(proposed_sample, cost)

        # determine acceptance probability
        a = mh_acceptance_prob(current_logpdf, proposed_logpdf, samples[ii-1,:], proposed_sample, proposal_logpdf, current_cov)
        
        # Accept or reject the sample
        u = np.random.rand()
        if a == 1 or u < a: # Accept
            samples[ii, :] = proposed_sample # Update the sample
            current_logpdf = proposed_logpdf # Update the logpdf
            covar[ii, :, :] = current_cov   # Update the covariance
            num_accept += 1
        else:
            # Check if delayed is true
            if delayed:
                # second level proposal
                delayed_cov = current_cov*gamma
                proposed_sample2 = sampler(samples[ii-1, :], delayed_cov)
                proposed2_logpdf, cost = target_logpdf(proposed_sample2,cost)

                # Accept or reject the second sample based on delayed acceptance probability
                a_delay = delay_acceptance_prob(current_logpdf, proposed_logpdf, proposed2_logpdf, samples[ii-1,:], proposed_sample, proposed_sample2, proposal_logpdf, delayed_cov)
                
                # Accept or reject the sample based on delayed acceptance prob
                u = np.random.rand()
                if a_delay == 1 or u < a_delay: #accept
                    samples[ii, :] = proposed_sample2
                    current_logpdf = proposed2_logpdf
                    covar[ii, :, :] = delayed_cov
                    num_accept += 1
                else: # reject
                    samples[ii, :] = samples[ii-1, :]
                    covar[ii, :, :] = current_cov 
            
            else: # if delayed is false reject
                samples[ii, :] = samples[ii-1, :]
                covar[ii, :, :] = current_cov 
        
        # Check if adaptive is true
        if adaptive:
            # Update the sample mean every iteration!
            xbar_minus = current_mean.copy()
            new_mean = (current_mean*ii + samples[ii, :]) / (ii+1)

            # Start adapting covariance after k0 steps
            if ii >= k0:
                k = ii
                x = samples[ii, :]
                xk = x[:, np.newaxis]
                xbark = new_mean[:, np.newaxis]
                xbark_minus = xbar_minus[:, np.newaxis]
                current_cov = (k-1)/k * current_cov + sd/k * \
                              (k * np.multiply(xbar_minus.T, xbark_minus) - 
                               (k+1) * np.multiply(xbark.T, xbark) + np.multiply(xk.T, xk) + 
                               eps * np.eye(dim))
                current_mean = new_mean
                covar[ii, :, :] = current_cov
                
        if np.mod(ii,save_counter) == 0:
            print(f"-------------------In Iteration {ii}--------------")
            np.savetxt(f'{output_fname}/samples_{ii}.csv', samples[ii, :], delimiter=",")
            np.savetxt(f"{output_fname}/covariance_{ii}.csv", current_cov, delimiter=",")
            np.savetxt(f"{output_fname}/currentmean_{ii}.csv", current_mean, delimiter=",")
    return samples, covar, num_accept / float(num_samples-1), cost

def am_gas(starting_sample: np.ndarray, starting_cov: np.ndarray, num_samples: int, target_logpdf: callable, proposal_logpdf: callable, sampler: callable, output_fname: str, am_C: float, am_alpha: float, am_ar: float, am_k0: int, am_stop: int, am_lamb: float, cost: int = 0, save_counter: int = 100, print_counter: int = 1000) -> dict[np.ndarray, float, np.ndarray, int]:
    """AM algorithm with Global Adaptive Scaling (Algorithm 4 in Andrieu, Christophe, and Johannes Thoms. “A Tutorial on Adaptive MCMC.” Statistics and Computing 18, no. 4 (December 2008): 343–73. https://doi.org/10.1007/s11222-008-9110-y.)
    
    Inputs
    ------
    starting_sample: (d, ) the initial sample
    starting_cov: (d, d) starting proposal covariance
    num_samples: positive integer, the number of total samples
    target_logpdf: function(x) -> (float, int), logpdf of the target distribution and cost of the target logpdf
    proposal_logpdf: function (x, y) -> float, logpdf of proposing y if current sample is x
    proposal_sampler: function (x) -> y, generate a sample if you are currently at x
    output_fname: str, name of the file to save the samples and covariance
    kwargs: dict, additional arguments for the algorithm
            kwargs['k0']: int, Number of steps to wait until adaptation kicks in
            kwargs['C']: float, the scaling parameter for the stepsizes (gamma_i = C/i^alpha)
            kwargs['alpha']: float, the scaling parameter for the stepsizes (gamma_i = C/i^alpha) alpha in ((1+\lambda)^-1, 1]
            kwargs['target_acceptance']: float, the target acceptance rate
    cost: int, sequential cost of the target logpdf
    save_counter: int, save the samples and covariance every save_counter iterations (for restart)
    
    Returns
    -------
    output_dict: dict, dictionary containing the samples, covariance, acceptance ratio and cost
        output_dict['Samples']: (num_samples, d) array of samples
        output_dict['Covariance']: (num_samples, d, d) history of covariance matrix
        output_dict['accept_ratio']: ratio of proposed samples that were accepted
        output_dict['cost']: int, cost of the whole algorithm
    """

    if isinstance(starting_sample, float)==True or isinstance(starting_sample, int)==True:
        raise ValueError("Starting sample should be an array not a float or int")
    else:
        # Get dimension of the problem
        dim = starting_sample.shape[0]

    if not os.path.exists(output_fname):
        os.makedirs(output_fname)

    samples = np.zeros((num_samples, dim))
    covariance = np.zeros((num_samples, dim, dim))
    current_mean = np.zeros((num_samples, dim))
    lam = np.zeros(num_samples)

    samples[0, :] = starting_sample
    current_logpdf, cost = target_logpdf(samples[0, :],cost)

    covariance[0, :, :] = starting_cov
    current_mean[0, :] = starting_sample
    lam[0] = am_lamb#2.4**2/dim
    
    num_accept = 0

    # Adaptation parameters (hardcoded for now)
    eps = 1e-7
    stop = am_stop

    for ii in range(1, num_samples):

        if np.mod(ii,print_counter) == 0:
            print(f"-------------------In Iteration {ii}--------------")

        if ii == stop:
            print('Adaptive phase completed')
            start_idx = ii - int(0.2*num_samples)
            end_idx = ii-1
            lam[ii-1] = np.mean(lam[start_idx:end_idx])
            covariance[ii-1, :, :] = np.mean(covariance[start_idx:end_idx], axis=0)

        # Update gamma
        gamma = am_C / (ii**am_alpha)
        # gamma = 1.0/ii

        # propose
        # proposed_sample = sampler(samples[ii-1, :], lam[ii-1]*covariance[ii-1, :, :])
        eta = sampler(np.zeros(dim), covariance[ii-1, :, :])
        proposed_sample = samples[ii-1, :] + np.dot((lam[ii-1]), eta)
        proposed_logpdf, cost = target_logpdf(proposed_sample, cost)

        # determine acceptance probability
        a = mh_acceptance_prob(current_logpdf, proposed_logpdf, samples[ii-1,:], proposed_sample, proposal_logpdf, covariance[ii-1, :, :])
        
        # Accept or reject the sample
        u = np.random.rand()
        if a == 1 or u < a: # Accept
            samples[ii, :] = proposed_sample # Update the sample
            current_logpdf = proposed_logpdf # Update the logpdf
            num_accept += 1
        else: # Reject
            samples[ii, :] = samples[ii-1, :]

        # Make the parameter udpates
        current_mean[ii, :] = current_mean[ii-1, :] + gamma*(samples[ii, :] - current_mean[ii-1, :])
        if ii >= am_k0 and ii < stop:
            # ar = num_accept / ii
            lam[ii] = lam[ii-1]*np.exp(gamma*(a-am_ar))
            covariance[ii, :, :] = covariance[ii-1, :, :] + gamma*(np.outer(samples[ii, :] - current_mean[ii, :], samples[ii, :] - current_mean[ii, :]) - covariance[ii-1, :, :] + eps * np.eye(dim))
        else:
            covariance[ii, :, :] = covariance[ii-1, :, :]
            lam[ii] = lam[ii-1]

        if np.mod(ii,save_counter) == 0:
            print(f"-------------------In Iteration {ii}--------------")
            np.save(f'{output_fname}/samples_{ii}.npy', samples)
            np.save(f"{output_fname}/currentmean_{ii}.npy", current_mean)
            np.save(f"{output_fname}/covariance_{ii}.npy", covariance)
            np.save(f"{output_fname}/lambda_{ii}.npy", lam)

    output_dict = {
        'samples': samples,
        'mean': current_mean,
        'covariance': covariance,
        'lambda': lam,
        'ar': num_accept / float(num_samples-1),
        'cost': cost,
    }

    return output_dict

def twostage_acceptance_prob(current_logpdf_high, proposed_logpdf_high, current_logpdf_low, proposed_logpdf_low):
    """Compute the two stage accept-reject probability
    
    Inputs
    ------
    current_logpdf_high : float, logpdf at the current sample in the fine chain
    proposed_logpdf_high : float, logpdf at the proposed sample in the fine chain
    current_logpdf_low : float, logpdf at the current sample in the coarse chain
    proposed_logpdf_low : float, logpdf at the proposed sample in the coarse chain
    
    Returns
    -------
    acceptance probability
    """

    check = proposed_logpdf_high - current_logpdf_high + current_logpdf_low - proposed_logpdf_low
    if check < 0:
        return np.exp(check)
    else:
        return 1

def da_mcmc(mcmc_params: dict, target_logpdf_list: list, mcmc_iters: int, proposal: dict, adaptive_params: dict) -> dict[np.ndarray, float, np.ndarray, int]:
    """
    Delayed Acceptance MCMC Algorithm

    Inputs
    ------
    mcmc_params: dict, dictionary containing the parameters for the MCMC
        mcmc_params['dim']: int, dimension of the parameter space
        mcmc_params['starting_point']: np.array, starting point for the MCMC chain
        mcmc_params['output_dir']: str, output directory for saving the samples
        mcmc_params['cost']: int, starting cost for the MCMC
        mcmc_params['save_counter']: int, counter for saving the samples
        mcmc_params['print_counter']: int, counter for printing the samples

    target_logpdf_list: list, list of target logpdf functions for the MCMC

    mcmc_iters: int, number of MCMC iterations (Outer loop)

    proposal: dict, dictionary housing the proposal distribution for MCMC
        proposal['logpdf']: callable, logpdf function for the proposal distribution
        proposal['sampler']: callable, sampler function for the proposal distribution
        proposal['cov']: np.array, covariance matrix for the proposal distribution

    adaptive_params: dict, dictionary housing the adaptive parameters for MCMC
        adaptive_params['am_C']: float, value of the adaptive parameter C
        adaptive_params['am_alpha']: float, value of the adaptive parameter alpha
        adaptive_params['am_ar']: float, target acceptance ratio for the adaptive MCMC
        adaptive_params['am_k0']: float, starting iteration for adaptation
        adaptive_params['am_stop']: int, ending iteration for adaptation
        adaptive_params['am_lamb']: float, initial scaling parameter for the proposal distribution

    Returns
    -------
    output_dict: dict, dictionary containing the samples, covariance, acceptance ratio and cost
        output_dict['Samples']: (num_samples, d) array of samples
        output_dict['Covariance']: (num_samples, d, d) history of covariance matrix
        output_dict['accept_ratio']: ratio of proposed samples that were accepted
        output_dict['cost']: int, cost of the whole algorithm
    """

    # Unpack and initialize the MCMC parameters
    dim = mcmc_params['dim']
    starting_point = mcmc_params['starting_point']
    cost = mcmc_params['cost']
    save_counter = mcmc_params['save_counter']
    print_counter = mcmc_params['print_counter']
    output_dir = mcmc_params['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Unpack the proposal distribution
    proposal_logpdf = proposal['logpdf']
    proposal_sampler = proposal['sampler']
    proposal_cov = proposal['cov'] 

    # Unpack the adaptive parameters
    am_C = adaptive_params['am_C']
    am_alpha = adaptive_params['am_alpha']
    am_ar = adaptive_params['am_ar']
    am_k0 = adaptive_params['am_k0']
    am_stop = adaptive_params['am_stop']
    am_lamb = adaptive_params['am_lamb']

    # Some basic checks
    if isinstance(starting_point, float)==True or isinstance(starting_point, int)==True:
        raise ValueError("Starting sample should be an array not a float or int")
    else:
        # Get dimension of the problem
        assert starting_point.shape[0] == dim, "Starting point and dimension do not match"

    assert len(target_logpdf_list) == 2, "Only two (high and low) target logpdfs are supported for now"

    target_logpdf_low = target_logpdf_list[0]
    target_logpdf_high = target_logpdf_list[1]

    samples = np.zeros((mcmc_iters, dim))
    covariance = np.zeros((mcmc_iters, dim, dim))
    current_mean = np.zeros((mcmc_iters, dim))
    lam = np.zeros(mcmc_iters)

    samples[0, :] = starting_point
    current_logpdf_low, cost = target_logpdf_low(samples[0, :], cost)
    current_logpdf_high, cost = target_logpdf_high(samples[0, :], cost)

    covariance[0, :, :] = proposal_cov
    current_mean[0, :] = starting_point
    lam[0] = am_lamb
    
    accept_low = np.zeros(mcmc_iters)
    accept_high = np.zeros(mcmc_iters)

    # Adaptation parameters (hardcoded for now)
    eps = 1e-7
    stop = am_stop

    a2 = 0.0

    for ii in range(1, mcmc_iters):

        if np.mod(ii,print_counter) == 0:
            print(f"-------------------In Iteration {ii}--------------")

        if ii == stop:
            print('Adaptive phase completed')
            start_idx = ii - int(0.2*mcmc_iters)
            end_idx = ii-1
            lam[ii-1] = np.mean(lam[start_idx:end_idx])
            covariance[ii-1, :, :] = np.mean(covariance[start_idx:end_idx], axis=0)

        # Update gamma
        gamma = am_C / (ii**am_alpha)

        # propose
        eta = proposal_sampler(np.zeros(dim), covariance[ii-1, :, :])
        proposed_sample = samples[ii-1, :] + np.dot((lam[ii-1]), eta)
        proposed_logpdf_low, cost = target_logpdf_low(proposed_sample, cost)

        # determine acceptance probability
        a1 = mh_acceptance_prob(current_logpdf_low, proposed_logpdf_low, samples[ii-1,:], proposed_sample, proposal_logpdf, covariance[ii-1, :, :])
        
        # Accept or reject the sample based on the low fidelity model
        u = np.random.rand()
        if a1 == 1 or u < a1: # Accept
            accept_low[ii] = 1

            proposed_logpdf_high, cost = target_logpdf_high(proposed_sample, cost)
            a2 = twostage_acceptance_prob(current_logpdf_high, proposed_logpdf_high, current_logpdf_low, proposed_logpdf_low)
            v = np.random.rand()

            if a2 == 1 or v < a2: 
                accept_high[ii] = 1
                samples[ii, :] = proposed_sample
                current_logpdf_low = proposed_logpdf_low
                current_logpdf_high = proposed_logpdf_high
        
            else: 
                samples[ii, :] = samples[ii-1, :]
        else:
            samples[ii, :] = samples[ii-1, :]

        # Make the parameter udpates
        current_mean[ii, :] = current_mean[ii-1, :] + gamma*(samples[ii, :] - current_mean[ii-1, :])
        if ii >= am_k0 and ii < stop:
            # ar = num_accept / ii
            lam[ii] = lam[ii-1]*np.exp(gamma*(a2-am_ar))
            covariance[ii, :, :] = covariance[ii-1, :, :] + gamma*(np.outer(samples[ii, :] - current_mean[ii, :], samples[ii, :] - current_mean[ii, :]) - covariance[ii-1, :, :] + eps * np.eye(dim))
        else:
            covariance[ii, :, :] = covariance[ii-1, :, :]
            lam[ii] = lam[ii-1]

        if np.mod(ii,save_counter) == 0:
            print(f"-------------------In Iteration {ii}--------------")
            np.save(f'{output_dir}/samples_{ii}.npy', samples)
            np.save(f"{output_dir}/currentmean_{ii}.npy", current_mean)
            np.save(f"{output_dir}/covariance_{ii}.npy", covariance)
            np.save(f"{output_dir}/lambda_{ii}.npy", lam)

    output_dict = {
        'samples': samples,
        'mean': current_mean,
        'covariance': covariance,
        'lambda': lam,
        'ar': np.sum(accept_high) / float(mcmc_iters-1),
        'accept_low': accept_low,
        'accept_high': accept_high,
        'cost': cost,
    }

    return output_dict

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
    
