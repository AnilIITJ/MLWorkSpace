import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    return np.tanh(x)

# Generate a range of values for plotting
x = np.linspace(-5, 5, 100)

# Apply each activation function to these values
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Create a plot for each activation function
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label="Sigmoid", color='blue')
plt.title("Sigmoid Function")
plt.grid()

# ReLU
plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label="ReLU", color='orange')
plt.title("ReLU Function")
plt.grid()

# Leaky ReLU
plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu, label="Leaky ReLU", color='green')
plt.title("Leaky ReLU Function")
plt.grid()

# Tanh
plt.subplot(2, 2, 4)
plt.plot(x, y_tanh, label="Tanh", color='red')
plt.title("Tanh Function")
plt.grid()

plt.tight_layout()
plt.show()

