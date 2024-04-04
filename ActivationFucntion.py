import numpy as np

def relu(x):
    """
    Apply the ReLU activation function.
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """
    Apply the Leaky ReLU activation function.
    """
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    """
    bug fix here
    Apply the Tanh activation function.
    """
    return np.tanh(x)

def main():
    random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

    relu_values = [relu(x) for x in random_values]
    leaky_relu_values = [leaky_relu(x) for x in random_values]
    tanh_values = [tanh(x) for x in random_values]

    print("ReLU, Leaky ReLU, and Tanh values for the given data points:")
    for x, relu_val, leaky_relu_val, tanh_val in zip(random_values, relu_values, leaky_relu_values, tanh_values):
        print(f"ReLU({x:.1f}) = {relu_val:.3f}, Leaky ReLU({x:.1f}) = {leaky_relu_val:.3f}, Tanh({x:.1f}) = {tanh_val:.3f}")

if __name__ == "__main__":
    main()
