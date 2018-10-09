import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import os



# Data set in csv file
DATA_SET_CSV = "assets/datasets/train.csv"

if not os.is_file(DATA_SET_CSV):
      print("The file is missing...")

# Read data from csv file.
data = pd.read_csv("assets/datasets/train.csv").as_matrix()
classifier = DecisionTreeClassifier()

xtrain = data[0:21000, 1:]
train_label = data[0:21000, 0]

classfier.fit(xtrain, train_label)

# testing
xtest = data[21000:, 1:]
actual_label = data[21000:, 0]

def main(number):
    digit = xtest[number]
    digit.shape = (28, 28)

    # visualize the digit
    plt.imshow(255-digit, cmap='gray')
    make_prediction = classifier.predict( xtest[number] )
    return make_prediction


if __name__ == __main__:
    number = int(input("Enter a number: "))
    main(number)
