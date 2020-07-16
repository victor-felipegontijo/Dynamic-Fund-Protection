import numpy as np
from Dynamic_Fund_Protection import DFP_BSPricing_Formula
from Dynamic_Fund_Protection import DFP_BSPricing_Expectations
from Dynamic_Fund_Protection import DFP_BSPricing_MonteCarloDiscrete
from Dynamic_Fund_Protection import DFP_BSPricing_MonteCarloBB
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
checks_per_day = 3
#_____________________________________________#

# Code parameters
dt = 1/(365*checks_per_day)
size = int(years/dt) + 1
checking_instants = np.array(range(0,size))*dt
#_____________________________________________#

## Model Parameters - Geometric Brownian Motion

# Volatility
sigma = 0.2
#_____________________________________________#

n_simulations = 1000000

formula_value = DFP_BSPricing_Formula(S_0, 1, r, sigma, K, 0, years)
print()
print('Formula value: ' + str(formula_value))


integrals_value = DFP_BSPricing_Expectations(S_0, 1, r, sigma, K, 0, years)
print()
print('Integrals value: ' + str(integrals_value))

print()
DFP_BSPricing_MonteCarloBB(n_simulations, S_0, 1, r, sigma, K, years)

#print()
#DFP_BSPricing_MonteCarloDiscrete(checking_instants, n_simulations, S_0, 1, r, sigma, K)

