import simulate_discrete_variables as sdvar
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

# Given an array of probabilities,
# choose one of them and return it.
# Notice it returns index, not variable, so
# you don't have to send the whole x's array.
def inverse_transform(p, improved=True):
  if improved:
    p.sort(reverse=True)
  F, i = p[0], 0
  u = np.random.uniform(0, 1)
  while u > F:
    F, i = F + p[i + 1], i + 1
  return i

# Generates a variable X from a distribution
# we don't know how to generate a variable from,
# using a variable Y from a distribution we do know
# how to generate variables from.
def accept_reject(px, py, dist, *params):
  c, pi = 0, 0
  for i in range(len(px)):
    if px[i] / py[i] > c:
      c = px[i] / py[i]
      pi = px[i]
  if c < 1: # magic check xD
    c = 1.1 # magic assignment xD x2

  while True:
    Y = dist(*params)
    u = np.random.uniform(0, 1)
    if u < px[Y] / pi:
      return Y

# Given an array of probabilities, this method
# creates an array with k * px[i] positions
# for every x[i] value such that k * p[i] is an
# integer value.
# Notice it returns index, not the variable, so
# you don't have to send the whole x's array.
def urn(px):
  digits = 0
  for p in px:
    digits = max(len(str(p).split('.')[1]), digits)
  digits = 10**digits

  urn_array = []
  for i in range(len(px)):
    positions = int(digits * px[i])
    for _ in range(positions):
      urn_array.append(i)

  pos = sdvar.uniform(0, digits - 1)
  return urn_array[pos]

# Given two different distributions generate one of them
def composition(generate_x, generate_y, alpha):
  u = np.random.uniform(0, 1)
  if u < alpha:
    return generate_x()
  return generate_y()

# Given n different distributions generate one of them
def n_composition(alphas, generate_vars):
  # For better performance, reorder the probabilities
  # of each distribution of being generated, and then
  # generate them when u (the uniform) is lower than the
  # cummulative of the alphas.
  alphas, generate_vars = (list(t) for t in zip(*sorted(zip(alphas, generate_vars), reverse=True)))
  alphas = np.cumsum(alphas)
  u = np.random.uniform(0, 1)
  for i in range(len(alphas)):
    if u < alphas[i]:
      return generate_vars[i](), i

# Given n different distributions returns their sum
def n_sum(alphas, generate_vars):
  sim_x = 0
  for i in range(len(alphas)):
    sim_x += generate_vars[i]() * alphas[i]
  return sim_x

# Given n different distributions returns their prod
def n_prod(alphas, generate_vars):
  sim_x = 0
  for i in range(len(alphas)):
    sim_x *= generate_vars[i]() * alphas[i]
  return sim_x