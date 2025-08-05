import random
import math


def sigmoid(z):
    """
    Calculates the sigmoid activation function.
    Formula: 1 / (1 + e^-z)
    """
    return 1 / (1 + math.exp(-z))


def compute_network():
    """
    Builds and computes a 2-layer neural network with random values.
    """
    # --- 1. Generate Random Inputs ---
    # Generate 3 random input values between -1 and 1.
    inputs = [random.uniform(-1, 1) for _ in range(3)]

    # --- 2. Hidden Layer (Layer 1) ---
    # This layer has 2 neurons, each receiving 3 inputs.

    # Generate random weights for the 2 hidden neurons.
    # hidden_weights[i] contains the weights for the i-th neuron.
    # hidden_weights[i][j] is the weight from input j to neuron i.
    hidden_weights = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(2)]

    # Generate random biases for the 2 hidden neurons.
    hidden_biases = [random.uniform(-1, 1) for _ in range(2)]

    # Calculate the output of the hidden layer.
    hidden_outputs = []
    for i in range(2):  # Loop through each of the 2 neurons
        # Calculate the net input (z) for the current neuron
        net_input_hidden = (
            sum(inputs[j] * hidden_weights[i][j] for j in range(3)) + hidden_biases[i]
        )

        # Apply the sigmoid activation function
        neuron_output = sigmoid(net_input_hidden)
        hidden_outputs.append(neuron_output)

    # --- 3. Output Layer (Layer 2) ---
    # This layer has 1 neuron, receiving 2 inputs from the hidden layer.

    # Generate random weights from the 2 hidden neurons to the output neuron.
    output_weights = [random.uniform(-1, 1) for _ in range(2)]

    # Generate a random bias for the output neuron.
    output_bias = random.uniform(-1, 1)

    # Calculate the final output of the network.
    # The inputs are the outputs from the hidden layer.
    net_input_output = (
        sum(hidden_outputs[i] * output_weights[i] for i in range(2)) + output_bias
    )
    final_output = sigmoid(net_input_output)

    # --- 4. Print All Values ---
    print("--- Inputs ---")
    print(f"Inputs: {[round(x, 3) for x in inputs]}")
    print("\n--- Hidden Layer (Layer 1) ---")
    # Corrected the list comprehension syntax for printing the nested list.
    print(
        f"Hidden layer weights: [[round(w, 3) for w in row] for row in hidden_weights]"
    )
    print(f"Hidden layer biases: {[round(b, 3) for b in hidden_biases]}")
    print(f"Hidden outputs: {[round(o, 3) for o in hidden_outputs]}")
    print("\n--- Output Layer (Layer 2) ---")
    print(f"Output layer weights: {[round(w, 3) for w in output_weights]}")
    print(f"Bias: {round(output_bias, 3)}")
    print("\n--- Final Result ---")
    print(f"Final Output: {round(final_output, 3)}")


# --- Main execution block ---
if __name__ == "__main__":
    compute_network()
