import scipy.special
import math

# Returns P(X2_k >= x)
# where X2_k is chi square with k degrees of liberty
def chi_square_CDF(k, x):
  return 1 - scipy.special.gammainc(k/2, x/2)

# returns P(X = k) for a poisson distriburion
def poisson_PDF(l, k):
  return math.e**(-l) * l**k / math.factorial(k)

# returns P(X < k) for a poisson distriburion
# for P(X <= k) use k+1, and for P(X > k) use 1-(X <= k)
def poisson_CDF(l, k):
  e, area = math.e**(-l), 0
  for i in range(k):
    area += l**i / math.factorial(i)
  return e * area
