import numpy as np
import math
import matplotlib.pyplot as plt 
from statistics import stdev
from statistics import mean
from matplotlib.pyplot import figure
from scipy.stats import sem
from scipy.stats import norm
from random import choice
from string import ascii_lowercase

# Plots the graph of a process
def plot(title, y, dt=1, color_ = 'green', name_ = 'fig'+ ''.join(choice(ascii_lowercase) for i in range(5)) ):
    
    l= len(y)
    instants = np.array(range(0,l))*dt
    
    plt.figure(figsize=(15,8))

    plt.plot(instants, y, color= color_, linestyle='dashed', linewidth = 0.1, marker='o', markersize=0.5)
   
    plt.xlabel('Instants (Years)') 
    plt.ylabel('Value') 
  
    plt.title(title)
    plt.savefig(name_)

# Plots and superimpose the graph of three process
def plot_superimpose(title, name, y1, y2, y3, dt=1):

    l= len(y1)
    instants = np.array(range(0,l))*dt
    
    plt.figure(figsize=(15,8))

    plt.scatter(instants, y1, s=2, color='red')
    plt.scatter(instants, y2, s=2, color='green')
    plt.scatter(instants, y3, s=2, color='blue')
   
    plt.xlabel('Instants (Years)') 
    plt.ylabel('Value') 
  
    plt.title(title)
    plt.savefig(name)

# Generates a Geometric Brownian Motion path 
def Geometric_Brownian_Motion(S_0, mu, sigma, instants):
    
    l = len(instants)
    
    positions = [S_0]

    # Generate the increments
    for i in range(1,l):
        dt = instants[i] - instants[i-1]
        positions.append( positions[i-1]*math.exp( (mu - sigma*sigma/2 )*dt + sigma*np.random.normal(0, math.sqrt(dt)) ) )

    return positions

# Generates a path of K
def Gen_K(K, instants):
    
    l = len(instants)
    positions = [K]
    
    # Generate the increments
    for i in range(1,l):
        positions.append(K)
    return positions

# Takes a path of S and returns the F-correspondent path
def Gen_F(S,K):
    
    l = len(S)
    
    positions = [S[0]]

    most_neg = 100000
    Been_neg = False
    # Generate the increments
    for i in range(1,l):

        if (S[i] < K):
            if(Been_neg == False):
                Been_neg = True
            
            if (S[i] < most_neg):
                most_neg = S[i]
        
        if(Been_neg == False):
            positions.append(S[i])
        else:
            positions.append(S[i]*K/most_neg)
    return positions

# Black Scholes Closed Formula Pricing
def DFP_BSPricing_Formula(S_t, M_t, r, sigma, K, t, T):

    tau = T-t
    if (tau == 0):
        tau = 0.0001

    R = 2*r/(sigma*sigma)
    K_prime = K/M_t
    kappa = math.log(S_t/K_prime)

    d1 = (kappa + r*tau + 0.5*sigma*sigma*tau)/(sigma*math.sqrt(tau))
    d2 = (-kappa + r*tau + 0.5*sigma*sigma*tau)/(sigma*math.sqrt(tau))
    d3 = (-kappa - r*tau + 0.5*sigma*sigma*tau)/(sigma*math.sqrt(tau))

    s1 = S_t*( M_t*norm.cdf(d1) - 1)
    s2 = K/R*math.pow((K_prime/S_t),R)*norm.cdf(d2)
    s3 = (1 - 1/R)*K*math.exp(-r*tau)*norm.cdf(d3)

    return s1 + s2 + s3

# Black Scholes Expectation Analytic Calculation Pricing
### CDF of the minimum of a Geometric Brownian motion between t and t + tau
def F_x(mu_, sigma, tau, S_t, x):
    
    y = math.log(x/S_t)
    c = 2*mu_/(sigma*sigma)
    
    s1 = norm.cdf( (-mu_*tau + y)/(sigma*math.sqrt(tau)) )
    s2 = math.exp(y*c)*norm.cdf( (mu_*tau + y )/(sigma*math.sqrt(tau)) )
    
    return s1 + s2

### Returns the value of the integral: Phi(c1 + c2*y)*exp(-c3*y)*dy from -infinity to b
def Integral_Phi_exp(c1, c2, c3, b):

    c4 = c1 + c3/c2

    s1 = -norm.cdf(c1 + c2*b)/(c3*math.exp(c3*b))
    s2 = (1/c3)*math.exp( (c4*c4 - c1*c1)/2 )*norm.cdf(c4 + c2*b)

    return s1 + s2

### Returns the value of the integral: K*F(x)/(x^2)*dx from 0 to B via substution x = S_t*exp(y)
def Integral_F_over_x2(K, B, mu_, r, sigma, tau, S_t,):

    b = math.log(B/S_t)
    
    K1 = (mu_*tau)/(sigma*math.sqrt(tau))
    K2 = 1/(sigma*math.sqrt(tau))
    K3 = 2*mu_/(sigma*sigma)

    s1 = (K/S_t)*Integral_Phi_exp(-K1, K2, 1, b)
    s2 = (K/S_t)*Integral_Phi_exp(K1, K2, (1-K3), b)

    return s1 + s2

###
def DFP_BSPricing_Expectations(S_t, M_t, r, sigma, K, t, T):

    tau = T-t
    if (tau == 0):
        tau = 0.0001
    
    mu_ = r + (sigma*sigma)/2
    
    expectation2 = F_x(mu_, sigma, tau, S_t, (K/M_t))
    expectation1 = M_t*expectation2 + Integral_F_over_x2(K, (K/M_t), mu_, r, sigma, tau, S_t)

    s1 = S_t*(M_t - 1)
    s2 = S_t*expectation1
    s3 = S_t*M_t*expectation2

    return s1 + s2 - s3  

#Black Scholes - Monte-Carlo pricing methods
def DFP_BSDiscrete_Simulation(instants, S_t, M_t, r, sigma, K):

    l = len(instants)
    min_S = S_t
    S = S_t

    # Generate the increments
    for i in range(1,l):
        dt = instants[i] - instants[i-1]
        S = S*math.exp( (r - sigma*sigma/2 )*dt + sigma*np.random.normal(0, math.sqrt(dt)) ) 

        if(S < min_S):
            min_S = S
    
    FT = S*max(M_t, K/min_S)
    XT = FT - S

    tau = instants[l-1] - instants[0]
    discounting_factor = math.exp(-r*tau)
    
    return discounting_factor*XT


def DFP_BSBrownianBridge_Simulation(S_t, M_t, r, sigma, K, tau):

    S = S_t*math.exp( (r - sigma*sigma/2 )*tau + sigma*np.random.normal(0, math.sqrt(tau)) )
    u = np.random.uniform()
    
    w_t = math.log(S_t)
    w = math.log(S)

    w_min = 0.5*(w_t + w - math.sqrt( (w_t - w)*(w_t - w) - 2*sigma*sigma*tau*math.log(u) ))

    S_min = math.exp(w_min)

    FT = S*max(M_t, K/S_min)
    XT = FT - S

    discounting_factor = math.exp(-r*tau)
    return discounting_factor*XT

def DFP_BSPricing_MonteCarloDiscrete(instants, n_simulations, S_t, M_t, r, sigma, K):
    
    l = len(instants)
   
    results = []

    for i in range(0, n_simulations):
        results.append( DFP_BSDiscrete_Simulation(instants, S_t, M_t, r, sigma, K) )
    
    print('Mean obtained with Discrete Monte-Carlo (' + str(n_simulations) + ' simulations, each one with ' + str(l) + ' discrete steps): ' + str(mean(results)))
    print('Standard Deviation obtained with Discrete Monte-Carlo: (' + str(n_simulations) + ' simulations, each one with ' + str(l) + ' discrete steps): ' + str(stdev(results)))
    print('Standard Error obtained with Discrete Monte-Carlo: (' + str(n_simulations) + ' simulations, each one with ' + str(l) + ' discrete steps): ' + str(sem(results)))


#Black Scholes Monte-Carlo Brownian Bridge
def DFP_BSPricing_MonteCarloBB(n_simulations, S_t, M_t, r, sigma, K, tau):

    results = []

    for i in range(0, n_simulations):
        results.append( DFP_BSBrownianBridge_Simulation(S_t, M_t, r, sigma, K, tau) )
    
    print('Mean obtained with Brownian-Bridge Monte-Carlo (' + str(n_simulations) + ' simulations): ' + str(mean(results)))
    print('Standard Deviation obtained with Brownian-Bridge Monte-Carlo: (' + str(n_simulations) + ' simulations): ' + str(stdev(results)))
    print('Standard Error obtained with Brownian-Bridge Monte-Carlo: (' + str(n_simulations) + ' simulations): ' + str(sem(results)))
