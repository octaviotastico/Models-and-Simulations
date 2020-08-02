from calculate_continue_distributions import chi_square_CDF
from arrays import frequency

########## Chi-squared test ##########

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

# Simulates p-value for a pearson chi square
# test for a discrete random variable.
# If p-value is very similar to alpha, CREATE
# new m's (samples), and call this function
# multiple times. Count the proportion (how
# many) of the p-values are lower than alpha,
# and take a new desition (for example, if 60%
# of pvals are lower, then reject it)
def pearson_chi_squared_test(m, x, p, alpha):
  T = chi_squared_statistic(m, x, p)
  p_value = chi_square_CDF(len(x) - 1, T)
  return p_value, 'Reject H0, pval < alfa' if (p_value <= alpha) else 'Dont reject H0, pval > alfa'

# If we have any unknown parameter in our distribution,
# we have to substract the amount of them in the
# degrees of freedom on the chi squared cdf.
def pearson_chi_squared_test_unknown_params(m, x, p, alpha, unknown_params, N=None, n=0, agrupation=False):
  T = chi_squared_statistic_with_agrupation(N, n, x, p) if agrupation else chi_squared_statistic(m, x, p)
  p_value = chi_square_CDF(len(x) - 1 - unknown_params, T)
  return p_value, p_value <= alpha
