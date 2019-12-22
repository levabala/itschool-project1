from keras.datasets import mnist
import matplotlib.pyplot as plt
from math import *

from genetic import fit

(X_train, y_train), (X_test, y_test) = mnist.load_data()

amount = 1000

x_train_pack = X_train[:amount]
y_train_pack = y_train[:amount]

print("Fitting on {} images...".format(len(x_train_pack)))
decider = fit(x_train_pack, y_train_pack)

print_step = floor(len(X_test) / 100)
correct_hits = 0

for i in range(100):
  x = X_test[i]
  y = y_test[i]

  decision = decider(x)

  correct = y == decision
  correct_hits += correct

  print("{} is right but i'll say it's {} (Right: {})".format(y, decision, correct))

  if i % print_step == 0:
    print("{:3.0f}% done | {:3.0f}% correct ".format(
        i / len(X_test) * 100, (correct_hits + 1) / (i + 1) * 100))


for i in range(len(X_test)):
  x = X_test[i]
  y = y_test[i]

  decision = decider(x)

  correct = y == decision
  correct_hits += correct

  if i % print_step == 0:
    print("{:3.0f}% done | {:3.0f}% correct ".format(
        i / len(X_test) * 100, (correct_hits + 1) / (i + 1) * 100))
