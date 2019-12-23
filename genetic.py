from random import random, shuffle
from math import *
from functools import reduce
from collections import namedtuple
from typing import *

import random as rnd

import matplotlib.pyplot as plt


def fit(X_raw, Y, coeffs_count=6, population_size_initial=100, random_injection_each_round=0, best_select_amount=4, mutation_rate=0.3, mutation_coeff=0.1, iterations_max=50):
  print("Fitting with mutation_rate={} and mutation_coeff={}".format(
      mutation_rate, mutation_coeff))
  # plt.subplot(221)
  # plt.imshow(X_comp[0], cmap=plt.get_cmap('gray'))

  # plt.subplot(222)
  # plt.imshow(X_comp[1], cmap=plt.get_cmap('gray'))

  # plt.subplot(223)
  # plt.imshow(X_comp[2], cmap=plt.get_cmap('gray'))

  # plt.subplot(224)
  # plt.imshow(X_comp[3], cmap=plt.get_cmap('gray'))

  # plt.show()

  def compressImage(img, color_range_original=255):
    img_comp = [[0] * coeffs_count for i in range(coeffs_count)]

    initial_size = len(img)

    comp_coeff = initial_size / coeffs_count
    comp_around = ceil(comp_coeff / 2)

    for y in range(coeffs_count):
      row_comp = img_comp[y]
      original_y = floor(y * comp_coeff)

      for x in range(coeffs_count):
        original_x = floor(x * comp_coeff)

        start_x = int(max(original_x - comp_around, 0))
        end_x = int(min(original_x + comp_around, initial_size))

        start_y = int(max(original_y - comp_around, 0))
        end_y = int(min(original_y + comp_around, initial_size))

        pixel_lines_around = [row_original[start_x:end_x]
                              for row_original in img[start_y:end_y]]

        amount = (end_x - start_x) * (end_y - start_y)
        average_around = reduce(
            lambda acc, line: acc + sum(line), pixel_lines_around, 0) / amount

        average_around_fit_color_range = average_around * \
            (1 / color_range_original)

        row_comp[x] = average_around_fit_color_range

    return img_comp

  def processNumber(number, chromosome, img):
    decision = 0
    for y in range(len(img)):
      row = img[y]
      for x in range(len(img)):
        weight = chromosome[y * len(img) + x]
        value = img[y][x]

        decision += weight * value

    return decision

  def randomWeight():
    return random() * 2 - 1

  def randomChromosome():
    gens = list(map(lambda x: randomWeight(), [0] * coeffs_count ** 2))
    return gens

  def calcError(X, number, chromosome):
    error = 0
    for i in range(len(X)):
      sameNumber = Y[i] == number
      prediction = processNumber(number, chromosome, X[i])

      if sameNumber:
        if prediction < 0:
          error += prediction ** 2
      else:
        if prediction > 0:
          error += sqrt(prediction)

    return error

  def stringifyImg(img):
    lines = []
    for row in img:
      row_s = list(map(lambda val: val.ljust(4), map(str, row)))
      lines.append(" ".join(row_s))

    return "\n".join(lines)

  def crossover(chro1, chro2):
    cross_x = floor(len(chro1) / 2)

    chro1_left_changed = chro1[:cross_x + 1] + chro2[cross_x:]
    chro2_right_changed = chro2[:cross_x + 1] + chro1[cross_x:]

    return chro1_left_changed, chro2_right_changed

  def mutate(chro):
    return list(map(lambda v: v if rnd.random() > mutation_rate else max(min(v + randomWeight() * mutation_coeff, 1), -1), chro))

  def fitForNumber(X, number: int, population=[randomChromosome() for i in range(population_size_initial)], iters_left=iterations_max, error_min_previous=None):
    if iters_left == iterations_max:
      print("--- fitting number {}".format(number))

    errors = list(map(lambda chro: calcError(
        X, number, chro), population))
    errors_enum = sorted(enumerate(errors), key=lambda t: t[1])
    error_min = errors_enum[0][1] / len(X)
    incorrect_answers = reduce(
        lambda acc, error: acc + (1 if error > 0 else 0), errors, 0)

    print("{} ticks left, population_size: {}, error_min: {:2.4f}, delta: {:2.6f}".format(iters_left, len(population),
                                                                                          error_min, 0 if error_min_previous == None else error_min_previous - error_min))

    population_sorted = [population[t[0]] for t in errors_enum]

    if iters_left <= 0 or error_min < 10e-5:
      return population_sorted[0]

    chro_best = population_sorted[:best_select_amount]
    chro_best_mutated = list(map(mutate, chro_best))

    shuffle(chro_best)

    pairs_to_crossover = zip(chro_best[::2], chro_best[1::2])
    crossovered_pairs = list(
        map(lambda pair: crossover(pair[0], pair[1]), pairs_to_crossover))
    crossovered_chromosomes = [
        chro for pair in crossovered_pairs for chro in pair]

    population_new = crossovered_chromosomes + \
        chro_best + chro_best_mutated

    # population_new = chro_best + chro_best_mutated

    return fitForNumber(X, number, population_new, iters_left - 1, error_min)

  def compressAllImages(X_raw):
    print("Started images compressing...")

    X = []

    print_step = floor(len(X_raw) / 100)
    counter = 0
    for img in X_raw:
      X.append(compressImage(img))

      if counter % print_step == 0:
        print("{:3.0f}% images compressed".format(counter / len(X_raw) * 100))

      counter += 1

    return X

  X = compressAllImages(X_raw)
  chromosomes = [fitForNumber(X, i) for i in range(10)]

  def decide(img):
    img_comp = compressImage(img)
    weights = [processNumber(number, chro, img_comp)
               for number, chro in enumerate(chromosomes)]
    weights_enum = sorted(enumerate(weights), key=lambda t: t[1])

    return weights_enum[-1][0]

  print("Fitting done with mutation_rate={} and mutation_coeff={}".format(
      mutation_rate, mutation_coeff))

  return decide
