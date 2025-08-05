import random
import math


def sigmoid(z):
    """Calculates the sigmoid activation function."""
    # Clamping z to avoid overflow error with large inputs to math.exp()
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))


def relu(z):
    """Calculates the Rectified Linear Unit (ReLU) activation function."""
    return max(0, z)


def run_configurable_network():
    """
    Builds and computes a 2-layer neural network based on user specifications.
    """
    try:
        # --- 1. Get User Configuration ---
        num_inputs = int(input("Enter number of inputs: "))
        num_hidden_neurons = int(input("Enter number of hidden neurons: "))
        activation_choice = input("Enter activation (sigmoid/relu): ").lower()

        if activation_choice not in ["sigmoid", "relu"]:
            print("Invalid activation function. Please choose 'sigmoid' or 'relu'.")
            return

        # --- 2. Generate Network Components ---
        # Generate random input values
        inputs = [random.uniform(-1, 1) for _ in range(num_inputs)]

        # --- Hidden Layer ---
        # Generate weights: a list of lists (h x n)
        hidden_weights = [
            [random.uniform(-1, 1) for _ in range(num_inputs)]
            for _ in range(num_hidden_neurons)
        ]
        # Generate biases for the hidden layer
        hidden_biases = [random.uniform(-1, 1) for _ in range(num_hidden_neurons)]

        # --- Output Layer ---
        # Generate weights from hidden layer to the single output neuron
        output_weights = [random.uniform(-1, 1) for _ in range(num_hidden_neurons)]
        # Generate a single bias for the output neuron
        output_bias = random.uniform(-1, 1)

        # --- 3. Perform Forward Pass ---
        # --- Hidden Layer Calculation ---
        hidden_outputs = []
        activation_function = relu if activation_choice == "relu" else sigmoid

        for i in range(num_hidden_neurons):
            # Calculate net input (z) for the neuron
            net_input = (
                sum(inputs[j] * hidden_weights[i][j] for j in range(num_inputs))
                + hidden_biases[i]
            )
            # Apply the chosen activation function
            output = activation_function(net_input)
            hidden_outputs.append(output)

        # --- Output Layer Calculation ---
        # Calculate net input for the output neuron
        output_net_input = (
            sum(
                hidden_outputs[i] * output_weights[i] for i in range(num_hidden_neurons)
            )
            + output_bias
        )
        # The final output is always passed through a sigmoid function for this network
        final_output = sigmoid(output_net_input)

        # --- 4. Print All Values ---
        print("\n--- Inputs ---")
        print(f"Inputs: {[round(x, 3) for x in inputs]}")

        print("\n--- Hidden Layer ---")
        print(
            f"Hidden layer weights: [[round(w, 3) for w in row] for row in hidden_weights]"
        )
        print(f"Hidden biases: {[round(b, 3) for b in hidden_biases]}")
        print(
            f"Hidden outputs ({activation_choice.upper()}): {[round(o, 3) for o in hidden_outputs]}"
        )

        print("\n--- Output Layer ---")
        print(f"Output layer weights: {[round(w, 3) for w in output_weights]}")
        print(f"Bias: {round(output_bias, 3)}")

        print("\n--- Final Result ---")
        print(f"Final Output (Sigmoid): {round(final_output, 3)}")

    except ValueError:
        print("\nError: Invalid input. Please enter integer values for counts.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    run_configurable_network()
