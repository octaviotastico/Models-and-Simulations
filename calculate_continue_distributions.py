import scipy.special

# Returns P(X2_k >= x)
# where X2_k is chi square with k degrees of liberty
def chi_square_CDF(k, x):
  return 1 - scipy.special.gammainc(k/2, x/2)