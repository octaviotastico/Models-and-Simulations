import calculate_continue_distributions as dist
from arrays import frequency
import numpy as np

########## Chi-squared Pearson's test ##########

# Returns chi squared T statistic
def chi_squared_statistic(x, p, s=None, N=None):
  N, T, n = frequency(s), 0, len(s)
  for i in range(len(x)):
    T += (((N[x[i]] if x[i] in N.keys() else 0) - n*p[i])**2) / (n*p[i])
  return T

# Returns chi squared T statistic
def chi_squared_statistic_agrupation(N, n, x, p):
  T = 0
  for i in range(len(x)):
    T += (((N[x[i]] if x[i] in N.keys() else 0) - n*p[i])**2) / (n*p[i])
  return T

# Simulates p-value using Pearson's chi square
# test for a discrete random variable, using
# the T statistic created by chi_squared_statistics
# functions above. Should be equal to scipy.stats.chisquare
def pearson_chi_squared_test(x, p, n, s=None, N=None):
  # If p-value is very similar to alpha, CREATE new s's (samples), and call this function
  # multiple times. Count the proportion (how many) of the p-values are lower than alpha,
  # and take a new desition (for example, if 60% of pvals are lower, then reject it)
  if s:
    T = chi_squared_statistic(x, p, s)
  else:
    T = chi_squared_statistic_agrupation(N, n, x, p)
  return T, dist.chi_square_CDF(len(x) - 1, T)

# If we have any unknown parameter in our distribution,
# we have to substract the amount of them in the
# degrees of freedom on the chi squared cdf.
def pearson_chi_squared_test_unknown_params(s, x, p, alpha, unknown_params, N=None, n=0, agrupation=False):
  T = chi_squared_statistic_agrupation(N, n, x, p) if agrupation else chi_squared_statistic(x, p, s)
  p_value = dist.chi_square_CDF(len(x) - 1 - unknown_params, T)
  return p_value, p_value <= alpha

########## Kolmogorov-Smirnov Test ##########

# Calculates the empirical distribution for some
# array x of data taken from an experiment
def empirical_dist(x):
  Fe = frequency(x)
  for elem in Fe:
    Fe[elem] /= len(x)
  return np.insert(np.cumsum(list(Fe.values())), 0, 0, axis=0)

def get_dks(s, Fe, dist, *params):
  s.sort()
  for i in range(len(s)):
    F = dist(*params, s[i])
    d_original = max(Fe[i+1] - F, F - Fe[i], d_original)

# This function calculates the p-value of a sample s
# in which we already KNOW all its parameters.
# The Kolmogorov Smirnov test creates a 'd' statistic
# comparing the H0 distribution values with the empiric
# distribution ones, then it gets the proportion of how many
# of some {sims} new d_i statistics created with a uniform
# distribution are greather than the original d statistic
def kolmogorov_smirnov(s, sims, dist, *params):
  Fe = empirical_dist(s)
  d_original, p_value = 0, 0
  size = len(s)
  s = list(set(s))
  s.sort()
  for i in range(len(s)):
    F = dist(*params, s[i])
    d_original = max(Fe[i+1] - F, F - Fe[i], d_original)
  for i in range(sims):
    u = np.random.uniform(0, 1, size)
    u.sort()
    d_i = 0
    for i in range(size):
      d_i = max(Fe[i+1] - u[i], u[i] - Fe[i], d_i)
    if d_i >= d_original:
      p_value += 1
  return d_original, p_value/sims

# This function calculates the p-value of a sample s
# in which we DONT KNOW some of its parameters.

# Mucho viaje, probably wont do it, porque hacerlo para
# un caso en especifico es re facil, y no se si vale la pena :/

# def kolmogorov_smirnov_unknown_params(s, sims, alpha, dist, *params):
#   size = len(s)
#   p_val = kolmogorov_smirnov(s, sims, dist, *params)
#   if p_val <= alpha:

#     for i in range(sims):
#       sample = []
#       for i in range(size):
#         sample.append(dist(params))
#       unknown_params = get_unknown_params(dist, *params)

