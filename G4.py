import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

import numpy as np
import time
import math

### Guia 4

def g4_ex1(): # Coincidence
  coincidences = 0
  sims, success = 10000, 0
  cards = list(range(1, 101))
  for sim in range(sims):
    cards = arr.permutation(cards)
    for i in range(1, 101):
      if cards[i - 1] == i:
        coincidences += 1
      if i == coincidences == 10:
        success += 1
  print(f'G4 EX1 - {success} Successes')
  print(f'Total iterations: {sims}')
  print(f'Total amount of coincidences: {coincidences}')
  print(f'E(x) = {coincidences/sims}')

def g4_ex2(): # Montecarlo mean
  def f(x):
    return math.e ** (x / 10000)

  s1 = time.perf_counter_ns()
  exact_amount = 0
  for i in range(10001):
    exact_amount += f(i)
  e1 = time.perf_counter_ns()

  s2 = time.perf_counter_ns()
  aproximation = arr.montecarlo_mean(100, 10000, f)
  e2 = time.perf_counter_ns()

  print(f'G4 EX2')
  print(f'Exact amount {exact_amount} vs Montecarlo Mean {aproximation}')
  print(f'Time spent in exact amount {(e1 - s1)/1000000}ms vs time spent in Montecarlo Mean {(e2 - s2)/1000000}ms')

def g4_ex3(): # Uniform sum
  sims = 10000
  success = []
  for _ in range(sims):
    dices, throws = set(), 0
    while len(list(dices)) < 11:
      dices.add(sdvar.uniform(1, 6) + sdvar.uniform(1, 6))
      throws += 1
    success.append(throws)
  print('G4 EX3')
  print(f'Total iterations: {sims}')
  print(f'Success average: {arr.mu(success)}')
  print(f'N > 15: {arr.proportion(success, 15, ">")}, N < 10: {arr.proportion(success, 10, "<")}')

def g4_ex4(): # Accept Reject
  px = [ 0.15, 0.20, 0.10, 0.35, 0.20 ]
  py = [ ddist.binomial_PDF(4, 0.45, i) for i in range(5) ]

  sim_x = sdvar.accept_reject(px, py, sdvar.binomial, 4, 0.45)
  print(f'G4 EX4 - X Simulated: {sim_x}')

def g4_ex5(): # Accept Reject
  px = [ 0.11, 0.14, 0.09, 0.08, 0.12, 0.10, 0.09, 0.07, 0.11, 0.09 ]
  py = [ cdist.uniform_PDF(0, 100) for i in range(100) ]

  sim_x = sdvar.accept_reject(px, py, sdvar.uniform, 0, 9)
  print(f'G4 EX5 - X Simulated: {sim_x}')

g4_ex1()
g4_ex2()
g4_ex3()
g4_ex4()
g4_ex5()
