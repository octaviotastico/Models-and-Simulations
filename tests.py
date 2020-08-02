import calculate_continue_distributions as dist
from arrays import frequency
import numpy as np

########## Chi-squared Pearson's test ##########

# Returns chi squared T statistic
def chi_squared_statistic(m, x, p):
  N, T, n = frequency(m), 0, len(m)
  for i in range(len(x)):
    T += (((N[x[i]] if x[i] in N.keys() else 0) - n*p[i])**2) / (n*p[i])
  return T

# Returns chi squared T statistic
def chi_squared_statistic_with_agrupation(N, n, x, p):
  T = 0
  for i in range(len(x)):
    T += (((N[x[i]] if x[i] in N.keys() else 0) - n*p[i])**2) / (n*p[i])
  return T

# Simulates p-value for a Pearson chi square
# test for a discrete random variable.
# If p-value is very similar to alpha, CREATE
# new m's (samples), and call this function
# multiple times. Count the proportion (how
# many) of the p-values are lower than alpha,
# and take a new desition (for example, if 60%
# of pvals are lower, then reject it)
def pearson_chi_squared_test(m, x, p, alpha):
  T = chi_squared_statistic(m, x, p)
  p_value = dist.chi_square_CDF(len(x) - 1, T)
  return p_value, 'Reject H0, pval < alfa' if (p_value <= alpha) else 'Dont reject H0, pval > alfa'

# If we have any unknown parameter in our distribution,
# we have to substract the amount of them in the
# degrees of freedom on the chi squared cdf.
def pearson_chi_squared_test_unknown_params(m, x, p, alpha, unknown_params, N=None, n=0, agrupation=False):
  T = chi_squared_statistic_with_agrupation(N, n, x, p) if agrupation else chi_squared_statistic(m, x, p)
  p_value = dist.chi_square_CDF(len(x) - 1 - unknown_params, T)
  return p_value, p_value <= alpha

########## Kolmogorov-Smirnov Test ##########

def empirical_dist(x):
  Fe = frequency(x)
  for elem in Fe:
    Fe[elem] /= len(x)
  return np.insert(np.cumsum(list(Fe.values())), 0, 0, axis=0)

def kolmogorov_smirnov(m, sims, dist, *params):
  Fe = empirical_dist(m)
  d_original, p_value = 0, 0
  size = len(m)
  m = list(set(m))
  m.sort()
  for i in range(len(m)):
    F = dist(*params, m[i])
    d_original = max(Fe[i+1] - F, F - Fe[i], d_original)
  print('K statistic', d_original)
  for i in range(sims):
    u = np.random.uniform(0, 1, size)
    u.sort()
    d_i = 0
    for i in range(size):
      d_i = max(Fe[i+1] - u[i], u[i] - Fe[i], d_i)
    if d_i >= d_original:
      p_value += 1
  print('p-value', p_value/sims)
  return p_value/sims
