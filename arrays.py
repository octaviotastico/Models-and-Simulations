from simulate_variables import uniform

# Returns a new array of length n made up
# with elements taken from the given list.
# All with equall probabilities of being chosen.
def sub_array(list, n):
  new_list = []
  for i in range(n):
    pos = uniform(0, len(list) - 1)
    print(pos)
    new_list.append(list[pos])
  return new_list

# Returns the average of a given list.
def mu(list):
  mu = 0
  for elem in list:
    mu += elem
  return mu / len(list)

# Returns the variance of a given list.
def sigma(list):
  sigma = 0
  mean = mu(list)
  for elem in list:
    sigma += (elem - mean)**(2)
  return sigma / len(list)

# Returns the variance and
# the mean of a given list.
def sigma_mu(list):
  sigma = 0
  mean = mu(list)
  for elem in list:
    sigma += (elem - mean)**(2)
  return sigma / len(list), mean