import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

import numpy as np
import time
import math

def g5_ex1(): # Inverse Transform Method continuous function
  def F_inver(x):
    if 0 < x <= 1/16:
      return np.log(16 * x) / 4
    if 1/16 < x <= 1:
      return 4 * x - 1 / 4
    return 0
  u = np.random.uniform(0, 1)
  print(f'G5 EX1 - Simulated X: {F_inver(u)}')

def g5_ex2(): # Weibull Inverse Transform Method
  def inverse_transform_weibull(l, k, x):
    return l * ((-np.log(1 - x))**(1 / k))
  u = np.random.uniform(0, 1)
  sim_x = inverse_transform_weibull(1, 2, u)
  print(f'G5 EX2 - Simulated Weibull X: {sim_x}')
  for _ in range(9999):
    sim_x += inverse_transform_weibull(1, 2, u)
  print(f'G5 EX2 - Mean of X (10.000 sims): {sim_x / 10000}')


# g5_ex1()
# g5_ex2()