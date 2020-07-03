import numpy as np
import matplotlib.pyplot as plt 
from Dynamic_Fund_Protection import plot
from Dynamic_Fund_Protection import plot_superimpose
from Dynamic_Fund_Protection import Geometric_Brownian_Motion
from Dynamic_Fund_Protection import Gen_K
from Dynamic_Fund_Protection import Gen_F

## General Parameters of the Investment
#Investment Horizont - Years
years = 5

#Annual interest rate
r = 0.1

#Initial value of a single unit
S_0 = 2.5
#_____________________________________________#

## Dynamic Fund Protection Parameters
# K  -(constant barrier)
K = 2

# Number of times per day the portfolio is monitored 
checks_per_day = 10
#_____________________________________________#

# Code parameters
dt = 1/(365*checks_per_day)
size = int(years/dt) + 1
checking_instants = np.array(range(0,size))*dt
#_____________________________________________#

## Model Parameters - Geometric Brownian Motion
# Drift
mu = -0.04

# Volatility
sigma = 0.2
#_____________________________________________#

## Simulating a Geometric Brownian Motion path of $S$, over the chosen period
S = Geometric_Brownian_Motion(S_0, mu, sigma, checking_instants)
plot('Log-normal S path:', S, dt, 'red', 'Log-normal S path')
#_____________________________________________#

## Determining the correspondent path of $F$, with respect to the simulated path of $S$, over the chosen period
K_floor = Gen_K(K, checking_instants)
F = Gen_F(S,K)
plot('Log-normal S - F correspondent path:', F, dt, 'green', 'Log-normal S - F correspondent path')
#_____________________________________________#

## Comparison between $S$ and the portfolio $F$ with the Dynamic Fund Protection
plot_superimpose('Log-normal S path x DFP Portfolio F x K barrier', 'Log-normal S path x DFP Portfolio F x K barrier', S , F, K_floor, dt)
#_____________________________________________#

plt.show()