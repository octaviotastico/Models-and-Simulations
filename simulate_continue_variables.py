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

# Generates a normal variable
# using the accept_reject method
# (2 iterations aprox)
def normal_reject_method(mu, sigma):
  while True:
    Y1 = -np.log(np.random.uniform(0, 1))
    Y2 = -np.log(np.random.uniform(0, 1))
    if Y2 >= ((Y1 - 1)**2) / 2:
      if np.random.uniform(0, 1) < 1/2:
        return Y1 * sigma + mu
      return -Y1 * sigma + mu

# Generates a normal variable
# using the polar method
# (2 iterations aprox)
def normal_polar_method(mu, sigma):
  radius_cuad = -2 * log(np.random.uniform(0, 1))
  theta = 2 * np.pi * np.random.uniform(0, 1)
  X = radius_cuad**(1/2) * np.cos(theta)
  Y = radius_cuad**(1/2) * np.sin(theta)
  return X*sigma+mu, Y*sigma+mu

# Generates a variable X from a distribution
# we don't know how to generate a variable from,
# using a variable Y from a distribution we do know
# how to generate variables from.
def accept_reject(f, fi, dist, *params):
  while True:
    Y = dist(*params)
    u = np.random.uniform(0, 1)
    if u < f(Y) / fi:
      return Y