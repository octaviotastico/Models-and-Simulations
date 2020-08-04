import calculate_continue_distributions as dist
import arrays as arr
import numpy as np

########## Chi-squared Pearson's test ##########

# Returns chi squared T statistic
def cs_statistic(x, p, s=None, N=None):
  N, T, n = arr.frequency(s), 0, len(s)
  for i in range(len(x)):
    T += (((N[x[i]] if x[i] in N.keys() else 0) - n*p[i])**2) / (n*p[i])
  return T

# Returns chi squared T statistic
def cs_statistic_agrupation(N, n, x, p):
  T = 0
  for i in range(len(x)):
    T += (((N[x[i]] if x[i] in N.keys() else 0) - n*p[i])**2) / (n*p[i])
  return T

# Simulates p-value using Pearson's chi square
# test for a discrete random variable, using
# the T statistic created by chi_squared_statistics
# functions above. Should be equal to scipy.stats.chisquare
def pearson_chi_squared_test(x, p, n, s=None, N=None, unknown_params=0):
  # If p-value is very similar to alpha, CREATE new s's (samples), and call this function
  # multiple times. Count the proportion (how many) of the p-values are lower than alpha,
  # and take a new desition (for example, if 60% of pvals are lower, then reject it)
  if s:
    T = cs_statistic(x, p, s)
  else:
    T = cs_statistic_agrupation(N, n, x, p)
  return T, dist.chi_square_CDF(len(x) - 1 - unknown_params, T)


########## Kolmogorov-Smirnov Test ##########

# Calculates the empirical distribution for some
# array x of data taken from an experiment
def empirical_dist(x):
  Fe = arr.frequency(x)
  for elem in Fe:
    Fe[elem] /= len(x)
  return np.insert(np.cumsum(list(Fe.values())), 0, 0, axis=0)

def ks_statistic(s, Fe, dist, *params):
  s.sort()
  stat = 0
  for i in range(len(s)):
    F = dist(*params, s[i])
    stat = max(Fe[i+1] - F, F - Fe[i], stat)
  return stat

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
  d_original = ks_statistic(s, Fe, dist, *params)
  for i in range(sims):
    u = np.random.uniform(0, 1, size)
    u.sort()
    d_i = 0
    for i in range(size):
      d_i = max(Fe[i+1] - u[i], u[i] - Fe[i], d_i)
    if d_i >= d_original:
      p_value += 1
  return d_original, p_value/sims

def two_samples_recursive(n, m, r):
  if (n and m) or (not n and not m):
    if not n:
      return two_samples_recursive(0, m - 1, r)
    elif not m:
      return two_samples_recursive(n - 1, 0, r)
    else:
      return (n * two_samples_recursive(n - 1, m, r - n - m) + m * two_samples_recursive(n, m - 1, r)) / (n + m)
  elif n and not m:
    if r < 1:
      return 0
    else:
      return 1
  else: # m and not n
    if r < 0:
      return 0
    else:
      return 1

def range_pvalue_test(l1, l2):
  return 2 * min(
    two_samples_recursive(len(l1), len(l2), arr.ranges(l1, l2)),
    1 - two_samples_recursive(len(l1), len(l2), arr.ranges(l1, l2) - 1)
  )
