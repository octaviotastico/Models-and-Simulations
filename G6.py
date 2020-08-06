import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

import numpy as np
import time
import math

def g6_ex1(): # Create sample and calculate mean/std
  normals = [ scvar.normal_reject_method(0, 1) ]
  std = 0
  while len(normals) < 30 or (std / len(normals))**(1 / 2) > 1 / 10:
    X, Y = scvar.normal_box_muller(0, 1)
    normals += [ X, Y ]
    std = arr.sigma(normals, True)
  print(f'G6 EX1 - {len(normals)} normals generated, mean {arr.mu(normals)}, var {arr.sigma(normals)}')

def g6_ex1_alt(): # Create sample and calculate mean/std
  normals, mean, std = arr.sample_mean_and_variance_error(0.01, 30, scvar.normal_reject_method, 0, 1)
  print(f'G6 EX1 - {len(normals)} normals generated, mean {mean}, var {std}')

def g6_ex2(): # Montecarlo integration
  def f(x):
    return math.e**x / (2 * x)**(1 / 2)
  montecarlo = arr.montecarlo_integral(f, 10000)
  print(f'G6 EX2 - Integral: {montecarlo}')

  montecarlo = [ arr.montecarlo_integral(f, 1) ]
  std = 0
  while len(montecarlo) < 100 or (std / len(montecarlo))**(1 / 2) > 1 / 100:
    montecarlo += [ arr.montecarlo_integral(f, 1) ]
    std = arr.sigma(montecarlo, True)
  print(f'G6 EX1 - {len(montecarlo)} normals generated, mean {arr.mu(montecarlo)}, var {arr.sigma(montecarlo)}')

def g6_ex3_a():
  def f(x):
    return math.sin(x) / x
  montecarlo = lambda: arr.montecarlo_integral_a_b(f, 1, math.pi, 2 * math.pi)
  interval, mean, std = arr.sample_mean_and_variance_length(0.025, 0.002, 100, montecarlo)
  print(interval, mean, std)

def g6_ex3_b():
  def f(x):
    return 2 / (3 + x**4)
  montecarlo = lambda: arr.montecarlo_integral_0_inf(f, 1)
  interval, mean, std = arr.sample_mean_and_variance_length(0.025, 0.05, 100000, montecarlo)
  print(interval, mean, std)

def g6_ex3_c():
  def f(x):
    return 2 / (3 + x**4)
  montecarlo = lambda: arr.montecarlo_integral_0_inf(f, 1)

  z_alfa_2, L, N = 0.025, 0.05, 100000
  sample = montecarlo()
  mean, std, n = sample, 0, 1
  length = L / (2 * z_alfa_2)
  while n <= N or (std/n)**(1/2) > length:
    prev_mean, n = mean, n + 1
    sample = montecarlo()
    mean = arr.recursive_mu(mean, sample, n)
    std = arr.recursive_sigma(std, prev_mean, mean, n)
  interval = (
    mean - z_alfa_2 * (std/n)**(1/2),
    mean + z_alfa_2 * (std/n)**(1/2)
  )
  print(interval, mean, std)

def g6_ex4():
  sum_u, n = 0, 0
  while sum_u < 1:
    u = np.random.uniform(0, 1)
    sum_u, n = u + sum_u, n + 1
  print(f'G6 EX4 - {n} uniform variables were needed.')


# g6_ex1()
# g6_ex1_alt()
# g6_ex2()
# g6_ex3_a()
# g6_ex3_b()
# g6_ex3_c()
# g6_ex4()


