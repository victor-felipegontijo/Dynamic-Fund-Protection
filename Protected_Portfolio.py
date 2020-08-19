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
    
    positions = []

    max_ratio = 1
    # Generate the increments
    for i in range(0,l):

        ratio = K/S[i]
        
        if ratio > max_ratio:
            max_ratio = ratio

        positions.append(S[i]*max_ratio)
    return positions

# Takes a path of S and returns the F-correspondent path
def Gen_F2(S,K):
    
    l = len(S)
    
    N = max(1, K/S[0])
    positions = [max(K,S[0])]

    # Generate the increments
    for i in range(1,l):

        dN = max(K/S[i] - N , 0)
        N = N + dN
        positions.append(N*S[i])
        
    return positions