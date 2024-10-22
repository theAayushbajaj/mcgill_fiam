import numpy as np
import pandas as pd
from scipy.optimize import minimize

######import packages
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import ARCHInMean, GARCH
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time #for the timer

import cvxopt as opt
from cvxopt import blas, solvers


def main(
    weights,
    returns,
    selected_stocks,
    benchmark_df,
    lambda_=1.0,
    soft_risk=0.01
):
    """
    Maximizes mu^T w - (1/2) * risk_aversion * w^T Sigma w subject to w >= 0
    and 0<= sum(w) <= 1 and w^T Sigma w <= vol(benchmark)^2

    Args:
        weights (pd.DataFrame): DataFrame containing the weights of the asset,
                                All possible stocks (not just selected ones)
        posterior_cov (pd.DataFrame): Posterior covariance matrix of the selected stocks
        posterior_mean (pd.Series): Posterior mean of the selected stocks
        selected_stocks (list): List of selected stocks

    Returns:
        pd.DataFrame: DataFrame containing the weights of all the assets
        (not selected stocks will have 0 weight)
    """
    
    weights_opt = reinforcement_learning(returns)

    weights.loc[selected_stocks, "Weight"] = weights_opt


    return weights




def reinforcement_learning(df):
    # df_original = df.copy()
    # df = df.droplevel(0, axis=1)# remove 'price' from column name
    # df =  df.sort_index()
    Returns = df # calc return from (final price - initial price / initial price)
    mult = 100
    lReturns = mult*np.log(1+Returns) 

    #####MARKET PARAMETERS

    nsteps = lReturns.shape[0] #number of periods in the investment horizon
    nassets = lReturns.shape[1] #number of assets in the portfolio

    ###ESTIMATION OF GARCH PARAMETERS
    coeffs = []
    cond_vol = []
    std_resids = []
    models = []

    for asset in lReturns.columns:
        model = arch_model(lReturns[asset], mean='Constant', 
                        vol='GARCH', p=1, o=0, 
                        q=1).fit(update_freq=0, disp='off')
        #model = ARCHInMean(lReturns[asset], lags=0, volatility=GARCH())
        #model.fix([0,0.5])
        #model = model.fit(update_freq=0, disp='off')
        coeffs.append(model.params)
        cond_vol.append(model.conditional_volatility)
        std_resids.append(model.resid / model.conditional_volatility)
        models.append(model)
            
    coeffs_df = pd.DataFrame(coeffs, index=lReturns.columns) #coefficients
    # cond_vol_df = pd.DataFrame(cond_vol).transpose() \
    #                                     .set_axis(lReturns.columns, 
    #                                               axis='columns', 
    #                                               inplace=False) #volatilities
    # std_resids_df = pd.DataFrame(std_resids).transpose() \
    #                                         .set_axis(lReturns.columns, 
    #                                                   axis='columns', 
    #                                                   inplace=False) #residuals

    cond_vol_df = pd.DataFrame(cond_vol).transpose() \
                                        .set_axis(lReturns.columns, 
                                                axis='columns'
                                                ) #volatilities
    std_resids_df = pd.DataFrame(std_resids).transpose() \
                                            .set_axis(lReturns.columns, 
                                                    axis='columns'
                                                    ) #residuals

    R = std_resids_df.transpose() \
                    .dot(std_resids_df) \
                    .div(len(std_resids_df)) #correlation matrix

    sigma0 = coeffs_df["omega"]/(1-coeffs_df["alpha[1]"]-coeffs_df["beta[1]"]) #initial log-return standard deviation for the first period

    ######SIMULATION PARAMETERS
    theseed = 100 #random seed
    split = 0.85
    nstepsIS = int(split*nsteps)  # number of in-sample simulated episodes
    nstepsOOS = nsteps - nstepsIS # number of out-of-sample simulated episodes

    ReturnsIS = Returns[:nstepsIS]
    VolatilitiesIS = cond_vol_df[:nstepsIS]
    ReturnsOSS = Returns[nstepsIS:]
    VolatilitiesOSS = cond_vol_df[nstepsIS:]

    ######UTILITY FUNCTION PARAMETERS
    eta= 5 #risk aversion parameter eta used in the utility function

    ######SIMULATION PARAMETERS
    theseed=100 #random seed
    # mIS=10000 # number of in-sample simulated episodes
    # mOOS=10000 # number of out-of-sample simulated episodes
    mIS=100 # number of in-sample simulated episodes
    mOOS=100 # number of out-of-sample simulated episodes


    ############################################################################
    ########### SIMULATION OF STATE VARIABLE PATHS FOR EPISODES ################
    ############################################################################

    np.random.seed(theseed) #Set random seed for results to be reproducible


    # is = insample oos - out of sample


    #IS = in-sample to train the agent
    simullogreturnsIS = np.empty([mIS, nsteps, nassets]) #matrix storing simulated risky asset log-returns 
    simulvolatilitiesIS = np.empty([mIS, nsteps + 1, nassets])  #matrix storing simulated volatilities
    simulvolatilitiesIS[:,0,:] = sigma0 #initial volatility
    #OS = out-of-sample, to test the agent performance on new data
    simullogreturnsOOS = np.empty([mOOS, nsteps, nassets])
    simulvolatilitiesOOS = np.empty([mOOS, nsteps + 1, nassets])
    simulvolatilitiesOOS[:,0,:] = sigma0 #initial volatility

    #####Simulate log-returns and volatilities for the risky asset. 
    for jj in range(nsteps):
        
        #simulate the next log-returns
        shockIS = np.random.multivariate_normal(mean=np.zeros(nassets), cov=R, size=mIS)
        shockOOS = np.random.multivariate_normal(mean=np.zeros(nassets), cov=R, size=mOOS)
    
        simullogreturnsIS[:,jj,:] = np.array(coeffs_df["mu"]) + simulvolatilitiesIS[:,jj,:]*shockIS #in-sample set
        simullogreturnsOOS[:,jj,:] = np.array(coeffs_df["mu"]) + simulvolatilitiesOOS[:,jj,:]*shockOOS #out-of-sample set
    
        #simulate the next volatility
        simulvolatilitiesIS[:,jj+1,:] = np.sqrt(np.array(coeffs_df["omega"]) + np.array(coeffs_df["alpha[1]"]) * (shockIS**2) + np.array(coeffs_df["beta[1]"])*simulvolatilitiesIS[:,jj,:]**2)
        simulvolatilitiesOOS[:,jj+1,:] = np.sqrt(np.array(coeffs_df["omega"]) + np.array(coeffs_df["alpha[1]"]) * (shockOOS**2) + np.array(coeffs_df["beta[1]"])*simulvolatilitiesOOS[:,jj,:]**2)

    #############################################################################################
    ################# SOLVING THE PROBLEM WITH DETERMINISTIC POLICY GRADIENT ####################
    #############################################################################################
    ##########functions needed to within the learning program######

    ###learning parameters
    nepochs = 1 #number of epochs during training
    minibatchsize = 10 #number of paths fed into the minibatch for each parameter update
    learingrate=1 #parameter related to learning speed

    #storing estimates of performance after each epoch
    PerfvecIS = np.zeros([nepochs])
    PerfvecOOS = np.zeros([nepochs])

    PerfvecIS15 = np.zeros([nepochs])
    PerfvecOOS15 = np.zeros([nepochs])

    ##design matrix construction: each row of the matrix is associated with one of the episode and contains 
    ##the vector [1, sigma_t]
    def designmatf(volvec):
        designmat = np.zeros([np.size(volvec), 1 ])+1
        designmat= np.append(designmat, volvec, axis=1)
        return(designmat)



    ##########functions needed to within the learning program######

    ##turns a vector of volatilities into a vector of actions (i.e. applies the mapping f(sigma_t,beta) to sigma_t)
    def volstoaction(volmat, currparam):
        curractions = np.array(currparam[:nassets]) + np.array(currparam[nassets:])*volmat
        
        curractions = np.exp(-curractions)/np.sum(np.exp(-curractions),1)[:,None]
        # curractions = np.exp(-curractions)/np.sum(np.exp(-curractions), axis=1).to_numpy()[:,None]

        return(curractions)

    ##evaluates the performance of a given policy on a given set of epsidoes (i.e. computes the objective function 
    ##over a given set of episodes for the current set of updated policy parameters)
    def evalperf(returnpaths, volpaths, currparam, eta=eta):
    
        totweightedretvec = np.zeros([np.shape(returnpaths)[0], 1]) + 1

        for jj in range(nsteps):

            volmat = np.reshape(volpaths[:,jj,:], (np.shape(returnpaths)[0], nassets))
            retmat = np.reshape(returnpaths[:,jj,:], (np.shape(returnpaths)[0], nassets))

            #Action selection
            curractions = np.reshape(volstoaction(volmat,currparam),(np.shape(returnpaths)[0],nassets))

            #Multiplies the previous partial utility by (1+R_{P,t})^{1-eta}. 
            #After the full loop this will give realized values for G_0^{1-eta}
            totweightedretvec = totweightedretvec * np.reshape((np.sum(curractions * np.exp(retmat/mult), 1))**(1-eta),(np.shape(returnpaths)[0],1))

        cumulperf = np.mean(totweightedretvec) #average across episodes to approximate the expectation  
        return(cumulperf)
    

    ##estimate of the objective function (loss fucntion )gradient for a set of paths
    def estimgradient(returnpaths,volpaths,currparam, eta=eta):
    
        totweightedretvec = np.zeros([np.shape(returnpaths)[0],1 ])+1
        secondsumterm = np.zeros([np.shape(returnpaths)[0],nparam ])

        for jj in range(nsteps): #LOOP OVER ALL TIME STEPS (NOTE THE ALL EPISODES DONE AT ONCE)

            volmat = np.reshape(volpaths[:,jj,:],(np.shape(returnpaths)[0],nassets))
            retmat = np.reshape(returnpaths[:,jj,:],(np.shape(returnpaths)[0],nassets))

            curractions = np.reshape(volstoaction(volmat,currparam),(np.shape(returnpaths)[0],nassets)) #Actions selected

            portfoliogrossreturn = np.sum(curractions * np.exp(retmat/mult),1) 
            totweightedretvec = totweightedretvec * (portfoliogrossreturn[:,None])**(1-eta)

            #dactiondbeta=designmatsplines(volvec=volpaths[,jj])
            interm = volstoaction(volmat, currparam)
            temp  = np.eye(nassets)
            temp = np.tile(temp, (np.shape(returnpaths)[0], 1, 1))
            temp = np.transpose(temp, (1, 2, 0)) - (np.ones(nassets)[None].T)*interm.T[:,None]
            temp = -interm.T[None,:,:] * temp
            temp2 = temp * volmat.T
            dactiondbeta = np.concatenate([temp, temp2], -2)

            secondsumterm = secondsumterm + (np.sum((np.exp(retmat/mult)/portfoliogrossreturn[:,None]).T[:,None,:]*dactiondbeta, 0)).T

        gradi = (1-eta)*np.mean( np.tile(totweightedretvec,(1,nparam)) *secondsumterm, axis=0)
        return(gradi)
    
    ###########RUN THE LEARNING ALGORITHM############
    np.random.seed(theseed+1)

    ##Initialize policy parameters to ad hoc parameters
    initparam = np.array([-3*np.ones(nassets),2*np.ones(nassets)]).flatten()
    Policyparam = initparam
    nparam = np.size(Policyparam)

    niter = math.floor(mIS/minibatchsize) #number of parameter updates per epoch

    start = time.time() #start timer to count how much time it takes to run the learning process

    # learning loop over epochs
    for ee in range(nepochs): #loop over epochs
    
        for nn in range(niter): #loop over mini-batches within an epoch (each corresponding to one parameter update)
            
            #check indices of episodes within the current minibatch
            batchrowid = np.array(range(minibatchsize)) + nn*minibatchsize 
            
            #estimate the gradient of the objective function for the current minibatch
            currgradient=estimgradient(returnpaths=simullogreturnsIS[batchrowid,],volpaths=simulvolatilitiesIS[batchrowid,],currparam=Policyparam)
            #print(currgradient)
            #apply a parameter update based on the gradient descent approach
            Policyparam = Policyparam - learingrate*currgradient
            #print(Policyparam)
    
        #After each episode, evaluate performance over both the training and test set of episodes
        PerfvecIS[ee] = evalperf(returnpaths=simullogreturnsIS,volpaths=simulvolatilitiesIS,currparam=Policyparam)
        PerfvecOOS[ee] = evalperf(returnpaths=simullogreturnsOOS,volpaths=simulvolatilitiesOOS,currparam=Policyparam)


    end = time.time() #end timer
    timeelapse = end - start
    print("Running time in seconds: " +str(round(timeelapse,2))) #print the amount of time needed to go through all epochs
    PolGrad = np.array([PerfvecIS[nepochs-1], PerfvecOOS[nepochs-1]]) # risk aversion (it depends on the portfolio manager) ## its a performance measure
    # PolGrad15 = np.array([PerfvecIS15[nepochs-1], PerfvecOOS15[nepochs-1]])
    
    weights_history = [] # track portfolio weights though time

    def volstoaction(volmat, currparam):
        curractions = np.array(currparam[:nassets]) + np.array(currparam[nassets:])*volmat
        
        # curractions = np.exp(-curractions)/np.sum(np.exp(-curractions),1)[:,None]
        curractions = np.exp(-curractions)/np.sum(np.exp(-curractions), axis=1).to_numpy()[:,None]

        weights_history.append(curractions)  # Append current weights to the history

        return(curractions)
    
    #weights = volstoaction(cond_vol_df,Policyparam)
    total_returnPolGrad = np.sum(Returns * volstoaction(cond_vol_df,Policyparam),1)
    #total_returnPolGrad = total_returnPolGrad[back_test_window:]
    cumReturnPolGrad = np.concatenate([[0.], list(np.cumprod(1+np.array(total_returnPolGrad))-1)])

    print("RL")
    #print(weights)
    # return ptf_stats(cumReturnPolGrad, total_returnPolGrad, freq='D'), total_returnPolGrad, cumReturnPolGrad
    return np.array(weights_history[0].iloc[-1])

