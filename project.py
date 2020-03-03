import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
T = 1000
y = np.zeros(len(train_labels))
#print(len(test_labels))
#Sorting into two classes (-1 and 1)
for i in range(len(train_labels)):
    y[i] = -1 if train_labels[i] < 5 else 1


a,b,c = train_images.shape
X = np.zeros((a, b*c))
for i in range(len(train_images)):
    X[i] = train_images[i].flatten()
S, features = X.shape[0], X.shape[1]
lamb = 0.1
w = np.zeros(features)
choice = np.arange(1,S+1)
for t in range(0,T):
    i = np.random.choice(choice)
    eta = 1/(lamb*(t+1))
    x_val, y_val = X[i], y[i]
    value = w.dot(x_val)
    if y_val*value < 1:
    	w = (1 - eta*lamb)*w + eta * y_val * x_val
    else:
    	w = (1 - eta * lamb) * w
#print(w.shape)

#Predicting the classes of the test samples
a, b, c = test_images.shape
X_test = np.zeros((a, b*c))
for i in range(len(test_images)):
    X_test[i] = test_images[i].flatten()
test_matrix = X_test.dot(w)
count_correct = 0
filename = "output.txt"
outfile = open(filename, 'w')
for i in range(len(test_labels)):
    if test_matrix[i] < 0 and test_labels[i] < 5:
        outfile.write('correct\n')
        count_correct += 1
    elif test_matrix[i] > 0 and test_labels[i] >= 5:
        outfile.write('correct\n')
        count_correct += 1
    else:
        outfile.write('wrong\n')
accuracy = float(count_correct/len(test_images))
outfile.write(str(accuracy * 100))
outfile.close()



