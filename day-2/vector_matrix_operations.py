def add_vectors(v1, v2):
    """
    Adds two vectors element-wise.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        list: A new vector representing the sum of v1 and v2.
              Returns None if vectors have different lengths.
    """
    if len(v1) != len(v2):
        print("Error: Vectors must have the same length for addition.")
        return None
    return [v1[i] + v2[i] for i in range(len(v1))]

def dot_product(v1, v2):
    """
    Computes the dot product of two vectors.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        int or float: The dot product of v1 and v2.
                      Returns None if vectors have different lengths.
    """
    if len(v1) != len(v2):
        print("Error: Vectors must have the same length for dot product.")
        return None
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def are_orthogonal(v1, v2):
    """
    Checks if two vectors are orthogonal.
    Two vectors are orthogonal if their dot product is zero.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        bool: True if the vectors are orthogonal, False otherwise.
              Returns None if vectors have different lengths.
    """
    dp = dot_product(v1, v2)
    if dp is None:
        return None
    return dp == 0

def multiply_matrices(matrix_a, matrix_b):
    """
    Multiplies two matrices using nested loops.

    Args:
        matrix_a (list of lists): The first matrix.
        matrix_b (list of lists): The second matrix.

    Returns:
        list of lists: The resultant matrix after multiplication.
                       Returns None if matrices cannot be multiplied
                       (i.e., number of columns in A != number of rows in B).
    """
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    # Check if multiplication is possible
    if cols_a != rows_b:
        print(f"Error: Cannot multiply matrices. "
              f"Number of columns in A ({cols_a}) must equal "
              f"number of rows in B ({rows_b}).")
        return None

    # Initialize the result matrix with zeros
    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    # Perform matrix multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):  # or rows_b, since they are equal
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result_matrix

if __name__ == "__main__":
    # --- Vector Operations ---
    print("--- Vector Operations ---")
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [1, 0, 0]
    d = [0, 1, 0]
    e = [1, 2] # For testing error handling

    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Vector c: {c}")
    print(f"Vector d: {d}")

    # Add two vectors
    sum_ab = add_vectors(a, b)
    if sum_ab is not None:
        print(f"Sum of a and b: {sum_ab}")

    sum_ae = add_vectors(a, e) # Should show error

    # Compute the dot product
    dot_prod_ab = dot_product(a, b)
    if dot_prod_ab is not None:
        print(f"Dot Product of a and b: {dot_prod_ab}")

    dot_prod_cd = dot_product(c, d)
    if dot_prod_cd is not None:
        print(f"Dot Product of c and d: {dot_prod_cd}")

    dot_prod_ae = dot_product(a, e) # Should show error

    # Check if two vectors are orthogonal
    orthogonal_ab = are_orthogonal(a, b)
    if orthogonal_ab is not None:
        print(f"Are a and b orthogonal?: {orthogonal_ab}")

    orthogonal_cd = are_orthogonal(c, d)
    if orthogonal_cd is not None:
        print(f"Are c and d orthogonal?: {orthogonal_cd}")

    orthogonal_ae = are_orthogonal(a, e) # Should show error

    print("\n--- Matrix Multiplication ---")
    A = [[1, 2],
         [3, 4]]

    B = [[5, 6],
         [7, 8]]

    C = [[1, 2, 3],
         [4, 5, 6]]

    D = [[7, 8],
         [9, 10],
         [11, 12]]

    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Matrix C:\n{C}")
    print(f"Matrix D:\n{D}")

    # Test with A and B
    result_ab = multiply_matrices(A, B)
    if result_ab is not None:
        print(f"\nResult of A * B:\n{result_ab}")

    # Test with C and D
    result_cd = multiply_matrices(C, D)
    if result_cd is not None:
        print(f"\nResult of C * D:\n{result_cd}")

    # Test with incompatible matrices (A and C)
    print("\nAttempting to multiply A and C (incompatible sizes):")
    result_ac = multiply_matrices(A, C)
    if result_ac is None:
        print("As expected, matrices A and C cannot be multiplied.")