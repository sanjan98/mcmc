"""
Example script to test MHMCMC and its variants on a multivariate Gaussian or a banana function
"""

# Imports
import os
import mcmc.utils.mcmc_utils_and_plot as utils
import mcmc.main.mcmc as mc
import numpy as np
import matplotlib.pyplot as plt

# Main code
def main(target):

    num_samples = 20000
    dim = 2
    guess = np.random.randn(dim) # random location
    
    if target=='gaussian':
        gauss_mean = np.array([1, 2])
        gauss_cov = utils.build_cov_mat(1.0, 1.0, 0.5) # std, std, correlation
        target_logpdf = lambda x,co: (utils.lognormpdf(x, gauss_mean, gauss_cov), co+1)
        # fig, axs = utils.plot_bivariate_gauss(np.linspace(-3, 5, 100), np.linspace(-2,7,100), 
        #                                 gauss_mean, gauss_cov)
        # plt.show()        
    
    elif target=='banana':
        # utils.plot_banana()
        # target_logpdf = utils.banana_logpdf 
        target_logpdf = utils.log_banana 
    
    sd = (2.4**2/dim)
    # map, prop_cov = utils.laplace_approx(guess, target_logpdf)
    # prop_cov = sd*prop_cov
    # print(map, prop_cov)
    map = np.array([0,0])
    prop_cov = np.array([[1,0],[0,1]])
    prop_sampler = lambda x,cov: utils.normal_sample(x,cov) 
    prop_logpdf = lambda x,y,cov: utils.lognormpdf(x,y,cov)

    # Create a folder to save everything
    os.makedirs("results", exist_ok=True)

    # DRAM sampler
    # samples, covariance, ar, cost = mc.dram(map, prop_cov, num_samples, target_logpdf, prop_logpdf, 
    #                       prop_sampler, output_fname="results", delayed=False, save_counter=100000) 
    # print("Accepted Samples Ratio:", ar)
    # print("Cost:", cost)

    am_C = 10.0; am_alpha = 0.5; am_ar = 0.234; am_k0 = 100
    output_dict = mc.am_gas(map, prop_cov, num_samples, target_logpdf, prop_logpdf, prop_sampler, 'results', am_C, am_alpha, am_ar, am_k0, save_counter=100000)

    samples = output_dict['samples']
    print("Accepted Samples Ratio:", output_dict['ar'])
    print("Cost:", output_dict['cost'])

    # plot samples from posterior
    fig, axs, gs = utils.scatter_matrix([samples], labels=[r'$x_1$', r'$x_2$'], 
                                hist_plot=False, gamma=0.4)
    fig.set_size_inches(7,7)
    fig, axs, gs = utils.scatter_matrix([samples], labels=[r'$x_1$', r'$x_2$'], 
                                hist_plot=True, gamma=0.2,
                                    nbins=70)
    fig.set_size_inches(7,7)
    plt.savefig("dram_samples2.png")

    fig, axs = plt.subplots(2,1, figsize=(10,5))
    axs[0].plot(samples[:, 0], '-k')
    axs[0].set_ylabel(r'$x_1$', fontsize=14)
    axs[1].plot(samples[:, 1], '-k')
    axs[1].set_ylabel(r'$x_2$', fontsize=14)
    axs[1].set_xlabel('Sample Number', fontsize=14)
    plt.savefig("dram_samples_trace2.png")

    maxlag=500
    step=1
    lags, autolag = utils.autocorrelation(samples, maxlag=maxlag,step=step)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(lags, autolag[:, 0],'-o')
    axs[0].set_xlabel('lag')
    axs[0].set_ylabel('autocorrelation dimension 1')
    axs[1].plot(lags, autolag[:, 1],'-o')
    axs[1].set_xlabel('lag')
    axs[1].set_ylabel('autocorrelation dimension 2')
    plt.savefig("dram_samples_autocorrelation2.png")


if __name__ == "__main__":
    target = 'gaussian'
    # target = 'banana'
    main(target)
    
