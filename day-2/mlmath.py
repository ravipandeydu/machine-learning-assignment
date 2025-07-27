# mlmath.py

def dot_product(a, b):
    """
    Computes the dot product of two vectors.

    The dot product (also known as the scalar product) is an algebraic operation
    that takes two equal-length sequences of numbers (vectors) and returns
    a single number.

    Args:
        a (list or tuple): The first vector (list or tuple of numbers).
        b (list or tuple): The second vector (list or tuple of numbers).

    Returns:
        int or float: The dot product of vectors 'a' and 'b'.

    Raises:
        ValueError: If the input vectors have different lengths.

    Examples:
        >>> mlmath.dot_product([1, 2, 3], [4, 5, 6])
        32
        >>> mlmath.dot_product([1, 0, 0], [0, 1, 0])
        0
        >>> mlmath.dot_product([2, 3], [-1, 2])
        4
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length for dot product.")
    return sum(a[i] * b[i] for i in range(len(a)))

def matrix_multiply(A, B):
    """
    Multiplies two matrices.

    For matrix multiplication, the number of columns in the first matrix (A)
    must be equal to the number of rows in the second matrix (B).

    Args:
        A (list of lists): The first matrix. Each inner list represents a row.
        B (list of lists): The second matrix. Each inner list represents a row.

    Returns:
        list of lists: The resultant matrix after multiplication.

    Raises:
        ValueError: If matrices cannot be multiplied due to incompatible dimensions.
                    (i.e., number of columns in A != number of rows in B).

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> B = [[5, 6], [7, 8]]
        >>> mlmath.matrix_multiply(A, B)
        [[19, 22], [43, 50]]

        >>> C = [[1, 2, 3], [4, 5, 6]]
        >>> D = [[7, 8], [9, 10], [11, 12]]
        >>> mlmath.matrix_multiply(C, D)
        [[58, 64], [139, 154]]
    """
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Cannot multiply matrices. "
            f"Number of columns in A ({cols_a}) must equal "
            f"number of rows in B ({rows_b})."
        )

    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result_matrix[i][j] += A[i][k] * B[k][j]
    return result_matrix

def conditional_probability(events):
    """
    Calculates conditional probability P(A|B) using the formula:
    P(A|B) = P(A and B) / P(B)

    This function expects a dictionary containing the probabilities of
    the intersection of events and the probability of the conditioning event.

    Args:
        events (dict): A dictionary with the following keys and values:
                       'P(A and B)' (float): The probability of both events A and B occurring.
                       'P(B)' (float): The probability of event B occurring.

    Returns:
        float: The conditional probability P(A|B).

    Raises:
        ValueError: If 'P(B)' is zero, as division by zero is not allowed.
                    If required keys are missing from the input dictionary.
        TypeError: If input probabilities are not numeric.

    Examples:
        >>> # Example: P(Spam | Free) = P(Spam and Free) / P(Free)
        >>> # From problem statement:
        >>> # P(Spam and Free) = 120 / 1000 = 0.12
        >>> # P(Free) = 300 / 1000 = 0.3
        >>> data = {'P(A and B)': 0.12, 'P(B)': 0.3}
        >>> mlmath.conditional_probability(data)
        0.4

        >>> # Another example: P(Rain | Clouds)
        >>> # P(Rain and Clouds) = 0.4, P(Clouds) = 0.6
        >>> data2 = {'P(A and B)': 0.4, 'P(B)': 0.6}
        >>> round(mlmath.conditional_probability(data2), 4)
        0.6667
    """
    required_keys = ['P(A and B)', 'P(B)']
    if not all(key in events for key in required_keys):
        raise ValueError(f"Input dictionary must contain keys: {required_keys}")

    p_a_and_b = events['P(A and B)']
    p_b = events['P(B)']

    if not isinstance(p_a_and_b, (int, float)) or not isinstance(p_b, (int, float)):
        raise TypeError("Probabilities must be numeric (int or float).")

    if p_b == 0:
        raise ValueError("P(B) cannot be zero for conditional probability P(A|B).")

    return p_a_and_b / p_b