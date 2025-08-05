import math


def calculate_neuron_output():
    """
    Asks the user for inputs, weights, and a bias, then calculates
    the output of a single neuron using the sigmoid activation function.
    """
    try:
        # --- 1. Get Inputs from the User ---
        # Get input values x1 and x2 from a single line
        x1, x2 = map(float, input("Enter x1, x2: ").split())

        # Get weight values w1 and w2 from a single line
        w1, w2 = map(float, input("Enter w1, w2: ").split())

        # Get the bias value
        bias = float(input("Enter bias: "))

        # --- 2. Calculate the Net Input (z) ---
        # The formula is the weighted sum of inputs plus the bias.
        z = (x1 * w1) + (x2 * w2) + bias

        # --- 3. Calculate the Output using Sigmoid Activation Function ---
        # The sigmoid function squashes the output to a range between 0 and 1.
        # Formula: sigmoid(z) = 1 / (1 + e^-z)
        output = 1 / (1 + math.exp(-z))

        # --- 4. Display the Result ---
        # Print the final output rounded to three decimal places.
        print(f"Neuron output: {output:.3f}")

    except ValueError:
        # Handle cases where the user enters non-numeric input
        print(
            "\nError: Invalid input. Please make sure you enter numbers separated by a space."
        )
    except IndexError:
        # Handle cases where the user doesn't enter two values where required
        print(
            "\nError: Please enter two values for inputs and weights, separated by a space."
        )


# --- Main execution block ---
if __name__ == "__main__":
    calculate_neuron_output()
