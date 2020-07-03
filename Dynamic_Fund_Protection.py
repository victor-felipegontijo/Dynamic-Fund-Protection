import numpy as np
import math
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
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
