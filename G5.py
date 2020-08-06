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

def g5_ex3_comp(): # Generate composition of N random variables
  alphas = [ 0.5, 0.3, 0.2 ]
  generate_vars = [
    lambda: scvar.exponential(3),
    lambda: scvar.exponential(5),
    lambda: scvar.exponential(7)
  ]
  sim_var, dist = sdvar.n_composition(alphas, generate_vars)
  print(f'G5 EX3 - Simulated X: {sim_var}, from distribution: {dist}')

  sim_vars, dists = [0]*3, [0]*3
  for _ in range(10000):
    sim_var, dist = sdvar.n_composition(alphas, generate_vars)
    sim_vars[dist] += sim_var
    dists[dist] += 1
  sim_vars = [ a / 10000 for a in sim_vars]

  print(
    f'After simulating 10000 variables, the composition method simulated:\n\
    {dists[0]} variables of type 0, with mean {sim_vars[0]},\n\
    {dists[1]} variables of type 1, with mean {sim_vars[1]},\n\
    {dists[2]} variables of type 2, with mean {sim_vars[2]}.'
  )

def g5_ex3_sum():
  alphas = [ 0.5, 0.3, 0.2 ]
  generate_vars = [
    lambda: scvar.exponential(3),
    lambda: scvar.exponential(5),
    lambda: scvar.exponential(7)
  ]
  sim_var = sdvar.n_sum(alphas, generate_vars)
  print(f'G5 EX3 Sum - Simulated X: {sim_var}')

  for _ in range(9999):
    sim_var += sdvar.n_sum(alphas, generate_vars)
  sim_var /= 10000

  print(f'Mean: {sim_var}')

# g5_ex1()
# g5_ex2()
# g5_ex3_comp()
# g5_ex3_sum()