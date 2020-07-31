import numpy as np

# Returns an uniform variable
# between the interval [a, b]
def uniform(a, b):
  u = np.random.uniform(0, 1)
  return int(u * (b - a + 1)) + a

# Returns a bernoulli with probability p
def bernoulli(p):
  u = np.random.uniform(0, 1)
  return 1 if u < p else 0

# Returns a geometric with probability p
def geometric(p):
  u = np.random.uniform(0, 1)
  return int(np.log(u)/np.log(1 - p)) + 1

# Creates an experiment of length n
# and returns 1 in the positions
# where an succes was made
def n_bernoulli(n, p):
  g = geometric(p)
  bernoullis = np.zeros(n, dtype=int)
  while g < n:
    bernoullis[g] = 1
    g += geometric(p)
  return bernoullis

# Returns a poisson variable
def poisson(lamda):
  p = F = np.exp(-lamda)
  for k in range(int(lamda)):
    p *= lamda / k
    F += p
  u, k = np.random.uniform(0, 1), int(lamda)
  if u < F:
    while u < F:
      F -= p
      p *= k / lamda
      k -= 1
    return k+1
  if u > F:
    while u > F:
      k += 1
      p *= lamda / k
      F += p
    return k

# Creates a binomial, and returns
# the k-th element
def binomial(n, p):
  k = 0
  F = prob = (1 - p)**(n)
  u = np.random.uniform(0, 1)
  while u >= F:
    prob *= p / (1 - p)
    prob *= (n - k) / (k + 1)
    F += prob
    k += 1
  return k