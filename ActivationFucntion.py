import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    return np.tanh(x)

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

relu_values = [relu(x) for x in random_values]
leaky_relu_values = [leaky_relu(x) for x in random_values]
tanh_values = [tanh(x) for x in random_values]

print("ReLU, Leaky ReLU, and Tanh values for the given data points:")
for x, relu_val, leaky_relu_val, tanh_val in zip(random_values, relu_values, leaky_relu_values, tanh_values):
    print(f"ReLU({x}) = {relu_val}, Leaky ReLU({x}) = {leaky_relu_val}, Tanh({x}) = {tanh_val}")
