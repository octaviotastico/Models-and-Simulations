from simulate_discrete_variables import uniform
import numpy as np

# Returns a new array of length n made up
# with elements taken from the given list.
# All with equall probabilities of being chosen.
def sub_array(list, n):
  new_list = []
  for i in range(n):
    pos = uniform(0, len(list) - 1)
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

# Calculates the average recursively.
def recursive_mu(prev_average, xn, n):
  return (prev_average * n + xn) / (n + 1)

# Returns the variance of a given list.
def sigma(s, sample=False):
  sigma = 0
  mean = mu(s)
  for elem in s:
    sigma += (elem - mean)**(2)
  return sigma / (len(s) - (1 if sample else 0))

# Calculates the variance recursively.
def recursive_sigma(prev_sigma, prev_average, average, n):
  return (1 - (1 / n)) * prev_sigma + (n + 1) * (prev_average * average)**2

# Returns the variance and
# the mean of a given list
def sigma_and_mu(s, sample=False):
  sigma = 0
  mean = mu(s)
  for elem in s:
    sigma += (elem - mean)**(2)
  return sigma / (len(s) - (1 if sample else 0)), mean

def proportion(list, x, condition):
  switch = {
    '<':  [ i for i in list if i <  x ],
    '<=': [ i for i in list if i <= x ],
    '==': [ i for i in list if i == x ],
    '>':  [ i for i in list if i >  x ],
    '>=': [ i for i in list if i >= x ]
  }
  return len(switch[condition])

# Estimate mean using Montecarlo
def montecarlo_mean(n, N, f):
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

# Calculates the mean of a distibution
# generating random values and updating it.
# Stops when completing minimum number of iterations,
# and when the variance is smaller than some error we want.
def sample_mean_and_variance_error(error, N, f, *params):
  sample = [f(*params)] # Because some methods, returns 2 variables
  mean, std = sample[0], 0
  while len(sample) < N or (std / len(sample))**(1 / 2) > error:
    prev_mean = mean
    sample += [f(*params)]
    mean = recursive_mu(mean, sample[-1], len(sample)) # Needs revision when sample increases in more than 1 variable per iteration
    std = recursive_sigma(std, prev_mean, mean, len(sample))
  return sample, mean, std

# Calculates the mean of a distibution
# generating random values and updating it.
# This KEEPS the array of variables, and returns it.
# Stops when completing minimum number of iterations
# and variance is smaller than some error.
def create_n_sample_variance_error(error, N, f, *params):
  mean = f(*params)
  variance, n = 0, 1
  while n <= N or (variance/n)**(1/2) > error:
    n += 1
    prev_mean = mean
    mean += (f(*params) - mean) / n
    variance = variance * (1 - 1/(n-1)) + n * (mean - prev_mean)**2
  return mean, variance

# Calculates the mean of a distibution
# generating random values and updating it.
# Stops when completing minimum number of iterations,
# and interval is smaller than some length we want.
def sample_mean_and_variance_length(z_alfa_2, L, N, f, *params, keep_sample=False):
  sample = [f(*params)]
  mean, std, n = sample[0], 0, 1
  length = L / (2 * z_alfa_2) # confianza (1-alfa)%
  while n <= N or (std/n)**(1/2) > length:
    prev_mean, n = mean, n + 1
    if keep_sample:
      sample += [f(*params)]
    else:
      sample = [f(*params)]
    mean = recursive_mu(mean, sample[-1], n)
    std = recursive_sigma(std, prev_mean, mean, n)
  interval = (
    mean - z_alfa_2 * (std/n)**(1/2),
    mean + z_alfa_2 * (std/n)**(1/2)
  )
  if keep_sample:
    return sample, interval, mean, std
  return interval, mean, std

# Calculates the mean of a distribution
# using the bootstrap method.
# Returns proportion of elements that
# has their mean between a and b
def bootstrap_mean(list, a, b, n):
  mean, bootstrap_mu, p = mu(list), 0, 0
  for _ in range(n):
    bootstrap = sub_array(list, len(list))
    actual_mu = mu(bootstrap)
    bootstrap_mu += actual_mu
    p += 1 if a <= (mean - actual_mu) <= b else 0
  return bootstrap_mu / n, p

# Calculates the variance of a distribution
# using the bootstrap method.
def bootstrap_variance(list, n):
  mean, result = mu(list), 0
  for _ in range(n):
    bootstrap = sub_array(list, len(list))
    bootstrap_mu = mu(bootstrap)
    result += (mean - bootstrap_mu)**2
  return result / n

# Returns a map containing how many
# times each element appears in a list
def frequency(list):
  m = {}
  for elem in list:
    if elem in m.keys():
      m[elem] += 1
    else:
      m[elem] = 1
  return m

# Returns range of two lists
def ranges(l1, l2, repeated=False, shuffle=False):
  if shuffle:
    l3 = permutation(l1 + l2)
  else:
    l3 = sorted(l1 + l2)
  range_l1 = 0
  if repeated:
    for x in l1:
      range_l1 += (len(l3) - l3[::-1].index(x) + l3.index(x) + 1) / 2
  else:
    for x in l1:
      range_l1 += l3.index(x) + 1

  return range_l1
