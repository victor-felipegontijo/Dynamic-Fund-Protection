import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Plots the graph of a process
def plot(title, y, dt=1, color_ = 'green'):
    
    l= len(y)
    instants = np.array(range(0,l))*dt
    
    plt.figure(figsize=(15,8))

    plt.plot(instants, y, color= color_, linestyle='dashed', linewidth = 0.1, marker='o', markersize=0.5)
   
    plt.xlabel('Instants (Years)') 
    plt.ylabel('Value') 
  
    plt.title(title)
    plt.show()

# Plots and superimpose the graph of three processes
def plot_superimpose(title, y1, y2, y3, dt=1):

    l= len(y1)
    instants = np.array(range(0,l))*dt
    
    plt.figure(figsize=(15,8))

    plt.scatter(instants, y1, s=2, color='red')
    plt.scatter(instants, y2, s=2, color='green')
    plt.scatter(instants, y3, s=2, color='blue')
   
    plt.xlabel('Instants (Years)') 
    plt.ylabel('Value') 
  
    plt.title(title)
    plt.show()

# Plots the Complementary Cumulative Distribution Function of the Running Maximum of a CEV process.
def plot_CCDF_CEV_RuningMaximum (x_0, tau, b, a, beta, liminf, limsup, n_points):

    from CEV_Maxima import G

    funct = lambda y: G(y, x_0, tau, b, a, beta)
    
    y = np.linspace(liminf, limsup, n_points)
    result = []
    for i in y:
        result.append(funct(i))

    plt.figure(figsize=(15,8))
    plt.xlabel('y') 
    plt.ylabel('Probability of Running Maximum over (0, tau) > y') 
  
    plt.title('CCDF of the Running Maximum of a CEV process: G(y, x_0, tau)')

    plt.plot(y, result)
    plt.show()