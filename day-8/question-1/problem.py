# 1. Import the PyTorch library
import torch

# --- Tensor Creation and Operations ---

print("--- Tensor Creation and Operations ---")

# 2. Create two tensors with random values from a standard normal distribution.
# A is a 3x2 tensor.
A = torch.randn(3, 2)
# B is a 2x3 tensor.
B = torch.randn(2, 3)

print(f"A:\n{A}\n")
print(f"B:\n{B}\n")

# 3. Perform the required computations.
# C is the result of the matrix multiplication of A and B.
# The '@' operator is used for matrix multiplication.
# The resulting shape is (3x2) @ (2x3) -> (3x3).
C = A @ B
print(f"C (Result of A @ B):\n{C}\n")

# D is the result of the element-wise addition.
# torch.ones_like(A) creates a tensor of ones with the same shape as A.
D = A + torch.ones_like(A)
print(f"D (Result of A + 1):\n{D}\n")

# 4. Check for GPU availability and move the result to the appropriate device.
# This line sets the device to 'cuda' (NVIDIA GPU) if available, otherwise it defaults to 'cpu'.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move tensor C to the selected device. The original tensor C remains on the CPU.
C_on_device = C.to(device)

# 5. Print the device information for the moved tensor.
print(f"C is on device: '{C_on_device.device}'")
