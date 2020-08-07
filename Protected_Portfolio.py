import numpy as np
import math

# Generates a Geometric Brownian Motion path 
def Geometric_Brownian_Motion(S_0, mu, sigma, instants):
    
    l = len(instants)
    
    positions = [S_0]

    # Generate the increments
    for i in range(1,l):
        dt = instants[i] - instants[i-1]
        positions.append( positions[i-1]*math.exp( (mu - sigma*sigma/2 )*dt + sigma*np.random.normal(0, math.sqrt(dt)) ) )

    return positions

# Generates a path of K(plane barrier) 
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

    most_neg = 1000000000000
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