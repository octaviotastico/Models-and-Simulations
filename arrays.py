from simulate_variables import uniform
import numpy as np

# Returns a new array of length n made up
# with elements taken from the given list.
# All with equall probabilities of being chosen.
def sub_array(list, n):
  new_list = []
  for i in range(n):
    pos = uniform(0, len(list) - 1)
    print(pos)
    new_list.append(list[pos])
  return new_list

# Returns a permutation of a list.
def permutation(list):
  for i in range(len(list)):
    index = uniform(i, len(list) - 1)
    list[index], list[i] = list[i], list[index]
  return list

# Returns the average of a given list.
def mu(list):
  mu = 0
  for elem in list:
    mu += elem
  return mu / len(list)

# Returns the variance of a given list.
def sigma(list):
  sigma = 0
  mean = mu(list)
  for elem in list:
    sigma += (elem - mean)**(2)
  return sigma / len(list)

# Returns the variance and
# the mean of a given list
def sigma_and_mu(list):
  sigma = 0
  mean = mu(list)
  for elem in list:
    sigma += (elem - mean)**(2)
  return sigma / len(list), mean

# Returns the variance of a sample
def sigma_sample(list):
  return (sigma(list) * len(n)) / (len(n) - 1)

# Returns the variance of a sample
# and the mean of a given list
def sigma_sample_and_mu(list):
  sigma, mu = sigma_and_mu(list)
  return (sigma * len(n)) / (len(n) - 1), mu

# Estimate mean using Montecarlo
def estimate_mu(n, N, f):
  mean = 0
  for _ in range(n):
    u = uniform(0, N)
    mean += f(u)
  return (mean / n) * N

# Estimates integral using Montecarlo
# for integrals between 0 and 1
def montecarlo_integral(f, n):
  integral = 0
  for _ in range(n):
    u = np.random.uniform(0, 1)
    integral += f(u)
  return integral / n

# Estimates integral using Montecarlo
# for integrals between a and b
def montecarlo_integral_a_b(f, n, a, b):
  integral = 0
  for _ in range(n):
    u = np.random.uniform(0, 1)
    integral += f(u * (b - a) + a)
  return (integral * (b - a)) / n

# Estimates integral using Montecarlo
# for integrals between a and b
def montecarlo_integral_0_inf(f, n):
  integral = 0
  for _ in range(n):
    u = np.random.uniform(0, 1)
    integral += f((1 / u) - 1) * (1/(u**2))
  return integral / n