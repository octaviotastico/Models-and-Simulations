import numpy as np
# Returns an uniform variable
# between the interval [a, b]
def uniform(a, b):
  u = np.random.uniform(0, 1)
  return int(u * (b - a + 1)) + a

def sub_array(list, n):
  new_list = []
  for i in range(n):
    pos = uniform(0, len(list) - 1)
    new_list.append(list[pos])
  return new_list

def mu(list):
  mu = 0
  for elem in list:
    mu += elem
  return mu / len(list)

def sigma(list):
  sigma = 0
  mean = mu(list)
  for elem in list:
    sigma += (elem - mean)**(2)
  return sigma / len(list)

def sigma_sample(list):
  return (sigma(list) * len(list)) / (len(list) - 1)

x = [1, 3]
prom = mu(x)
S = sigma_sample(x)

b1 = sub_array(x, 2)
var1 = sigma_sample(b1)

b2 = sub_array(x, 2)
var2 = sigma_sample(b2)

print(S, var1, var2, (var1+var2)/2)
