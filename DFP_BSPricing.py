import math
import numpy as np
from scipy.stats import sem
from scipy.stats import norm
from statistics import stdev
from statistics import mean

# 1) - Black Scholes DFP - Pricing from Closed Formula
def DFP_BSPricing_Formula(S_t, M_t, r, q, sigma, K, t, T):

    tau = T-t
    if (tau == 0):
        tau = 0.0001

    R = 2*(r-q)/(sigma*sigma)
    K_prime = K/M_t
    kappa = math.log(S_t/K_prime)

    d1 = (kappa + (r-q)*tau + 0.5*sigma*sigma*tau)/(sigma*math.sqrt(tau))
    d2 = (-kappa + (r-q)*tau + 0.5*sigma*sigma*tau)/(sigma*math.sqrt(tau))
    d3 = (-kappa - (r-q)*tau + 0.5*sigma*sigma*tau)/(sigma*math.sqrt(tau))

    s1 = S_t*( M_t*norm.cdf(d1) - 1)
    s2 = K/R*math.pow((K_prime/S_t),R)*norm.cdf(d2)
    s3 = (1 - 1/R)*K*math.exp(-(r-q)*tau)*norm.cdf(d3)

    return math.exp(-q*tau)*(s1 + s2 + s3)

################################################################################################################################################

# 2) - Black Scholes DFP - Pricing from Expectation Analytic Calculation 

## 2.1) - CDF of the running minimum of GBM between t and t + tau
def F_x(mu_, sigma, tau, S_t, x):
    
    y = math.log(x/S_t)
    c = 2*mu_/(sigma*sigma)
    
    s1 = norm.cdf( (-mu_*tau + y)/(sigma*math.sqrt(tau)) )
    s2 = math.exp(y*c)*norm.cdf( (mu_*tau + y )/(sigma*math.sqrt(tau)) )
    
    return s1 + s2

## 2.2) - Closed formula for the integral: Phi(c1 + c2*y)*exp(-c3*y)*dy from -infinity to b
def Integral_Phi_exp(c1, c2, c3, b):

    c4 = c1 + c3/c2

    s1 = -norm.cdf(c1 + c2*b)/(c3*math.exp(c3*b))
    s2 = (1/c3)*math.exp( (c4*c4 - c1*c1)/2 )*norm.cdf(c4 + c2*b)

    return s1 + s2

## 2.3) - Closed formula for the integral: K*F(x)/(x^2)*dx from 0 to B via substution x = S_t*exp(y)
def Integral_F_over_x2(K, B, mu_, r, sigma, tau, S_t,):

    b = math.log(B/S_t)
    
    K1 = (mu_*tau)/(sigma*math.sqrt(tau))
    K2 = 1/(sigma*math.sqrt(tau))
    K3 = 2*mu_/(sigma*sigma)

    s1 = (K/S_t)*Integral_Phi_exp(-K1, K2, 1, b)
    s2 = (K/S_t)*Integral_Phi_exp(K1, K2, (1-K3), b)

    return s1 + s2

## 2.4) - Black Scholes DFP pricing from the integrals above
def DFP_BSPricing_Expectations(S_t, M_t, r, q, sigma, K, t, T):

    tau = T-t
    if (tau == 0):
        tau = 0.0001
    
    mu_ = r - q + (sigma*sigma)/2
    
    s1 = M_t - 1
    s2 = Integral_F_over_x2(K, (K/M_t), mu_, r, sigma, tau, S_t)
    
    return math.exp(-q*tau)*S_t*(s1 + s2)

################################################################################################################################################

# 3) - Black Scholes DFP - Monte-Carlo pricing methods

#########

## 3.1) - Traditional Monte-Carlo

###  3.1.1) - Generates a single sample path and determines the payoff under discrete monitoring
def DFP_BSDiscrete_Simulation(instants, S_t, M_t, r, q, sigma, K):

    l = len(instants)
    min_S = S_t
    S = S_t

    # Generate the increments
    for i in range(1,l):
        dt = instants[i] - instants[i-1]
        S = S*math.exp( (r - q - sigma*sigma/2 )*dt + sigma*np.random.normal(0, math.sqrt(dt)) ) 

        if(S < min_S):
            min_S = S
    
    FT = S*max(M_t, K/min_S)
    XT = FT - S

    tau = instants[l-1] - instants[0]
    discounting_factor = math.exp(-r*tau)
    
    return discounting_factor*XT

###  3.1.2) - Determines the discrete-monitoring payoff under several simulation trials and calculates the empirical mean 
def DFP_BSPricing_MonteCarloDiscrete(instants, n_simulations, S_t, M_t, r, q, sigma, K):
    
    l = len(instants)
   
    results = []

    for i in range(0, n_simulations):
        results.append( DFP_BSDiscrete_Simulation(instants, S_t, M_t, r, q, sigma, K) )
    
    print('Mean obtained with Discrete Monte-Carlo (' + str(n_simulations) + ' simulations, each one with ' + str(l) + ' discrete steps): ' + str(mean(results)))
    print('Standard Deviation obtained with Discrete Monte-Carlo: (' + str(n_simulations) + ' simulations, each one with ' + str(l) + ' discrete steps): ' + str(stdev(results)))
    print('Standard Error obtained with Discrete Monte-Carlo: (' + str(n_simulations) + ' simulations, each one with ' + str(l) + ' discrete steps): ' + str(sem(results)))

#########

## 3.2) - Brownian-Bridge Monte-Carlo 

###  3.2.1) - Determines the payoff of a single sample path by simulating the final and minumum values
def DFP_BSBrownianBridge_Simulation(S_t, M_t, r, q, sigma, K, tau):

    S = S_t*math.exp( (r - q - sigma*sigma/2 )*tau + sigma*np.random.normal(0, math.sqrt(tau)) )
    u = np.random.uniform()
    
    w_t = math.log(S_t)
    w = math.log(S)

    w_min = 0.5*(w_t + w - math.sqrt( (w_t - w)*(w_t - w) - 2*sigma*sigma*tau*math.log(u) ))

    S_min = math.exp(w_min)

    FT = S*max(M_t, K/S_min)
    XT = FT - S

    discounting_factor = math.exp(-r*tau)
    return discounting_factor*XT

###  3.2.2) - Determine the "brownian-bridge" payoff under several simulation trials and calculates the empirical mean 
def DFP_BSPricing_MonteCarloBB(n_simulations, S_t, M_t, r, q, sigma, K, tau):

    results = []

    for i in range(0, n_simulations):
        results.append( DFP_BSBrownianBridge_Simulation(S_t, M_t, r, q, sigma, K, tau) )
    
    print('Mean obtained with Brownian-Bridge Monte-Carlo (' + str(n_simulations) + ' simulations): ' + str(mean(results)))
    print('Standard Deviation obtained with Brownian-Bridge Monte-Carlo: (' + str(n_simulations) + ' simulations): ' + str(stdev(results)))
    print('Standard Error obtained with Brownian-Bridge Monte-Carlo: (' + str(n_simulations) + ' simulations): ' + str(sem(results)))