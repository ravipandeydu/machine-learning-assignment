import numpy as np
import matplotlib.pyplot as plt

# --- 1. Activation Functions ---


def sigmoid(z):
    """Sigmoid activation function."""
    # Clip z to prevent overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def tanh(z):
    """Hyperbolic Tangent (Tanh) activation function."""
    return np.tanh(z)


def relu(z):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, z)


def leaky_relu(z, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(z > 0, z, z * alpha)


# --- 2. Forward Propagation ---


def forward_pass(inputs, weights, biases, activation_fn):
    """
    Performs a full forward pass through the network.

    Args:
        inputs (np.array): The initial input vector.
        weights (list): A list of weight matrices for each layer.
        biases (list): A list of bias vectors for each layer.
        activation_fn (function): The activation function to use.

    Returns:
        np.array: The final output of the network.
    """
    # The activation of the first layer is the input itself
    a = inputs

    # Loop through each layer of the network
    for i in range(len(weights)):
        # Calculate the net input (z) for the current layer
        z = np.dot(weights[i], a) + biases[i]

        # Apply the activation function
        # For the final layer, a linear activation (no change) is often used,
        # but here we apply the chosen activation to see its effect.
        a = activation_fn(z)

    return a


# --- 3. Main Simulation ---


def run_simulation(seed=42):
    """
    Generates a random network, runs the simulation, and plots the results.
    """
    np.random.seed(seed)
    print(f"Random Seed: {seed}\n")

    # --- a. Randomly Generate Network Structure ---
    num_inputs = np.random.randint(3, 7)
    num_hidden_layers = np.random.randint(1, 4)

    # Define the number of neurons in each layer
    layer_sizes = [num_inputs]
    for _ in range(num_hidden_layers):
        layer_sizes.append(np.random.randint(2, 6))
    layer_sizes.append(1)  # Single neuron in the output layer

    # --- b. Randomly Generate Network Parameters ---
    inputs = np.random.uniform(-10, 10, size=(num_inputs, 1))
    weights = []
    biases = []

    # Create weights and biases for each layer transition
    for i in range(len(layer_sizes) - 1):
        # Weight matrix shape: (neurons_in_current_layer, neurons_in_previous_layer)
        w = np.random.uniform(-1, 1, size=(layer_sizes[i + 1], layer_sizes[i]))
        # Bias vector shape: (neurons_in_current_layer, 1)
        b = np.random.uniform(-1, 1, size=(layer_sizes[i + 1], 1))
        weights.append(w)
        biases.append(b)

    # --- c. Print Network Structure ---
    print("--- Generated Network ---")
    print(f"- Input Features: {num_inputs} -> Values: {np.round(inputs.flatten(), 2)}")
    print(f"- Hidden Layers: {num_hidden_layers}")
    for i in range(num_hidden_layers):
        print(f"  - Layer {i+1}: {layer_sizes[i+1]} neurons")
    print(f"- Output Layer: {layer_sizes[-1]} neuron\n")

    # --- d. Perform Forward Pass for Each Activation ---
    activation_functions = {
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "ReLU": relu,
        "Leaky ReLU": leaky_relu,
    }

    final_outputs = {}
    print("--- Final Outputs ---")
    for name, func in activation_functions.items():
        output = forward_pass(inputs, weights, biases, func)
        final_outputs[name] = output.flatten()[0]
        print(f"- {name}: {np.round(output.flatten(), 3)}")

    # --- e. Plot the Results ---
    names = list(final_outputs.keys())
    values = list(final_outputs.values())

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    ax.set_ylabel("Final Output Value", fontsize=12)
    ax.set_title(
        "Comparison of Activation Functions on a Random Network", fontsize=16, pad=20
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom" if yval >= 0 else "top",
            ha="center",
            fontsize=11,
        )

    plt.tight_layout()
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # You can change the seed to generate a different network
    run_simulation(seed=42)
