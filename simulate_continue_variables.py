import numpy as np

# Returns an exponential variable
def exponential(lamda):
  u = np.random.uniform(0, 1)
  return - np.log(u) / lamda

# Returns an poisson variable
def poisson(lamda):
  x = 0
  cota = np.exp(-lamda)
  prod = np.random.uniform(0, 1)
  while prod >= cota:
    prod *= np.random.uniform(0, 1)
    x += 1
  return x

# Returns an gamma variable with param (1/lamda)
def gamma(n, lamda_inv):
  u = 1
  for _ in range(n):
    u *= np.random.uniform(0, 1)
  return -np.log(u) / lamda_inv
