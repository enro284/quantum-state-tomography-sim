# %% [markdown]
#  # Linear Inversion for Quantum State Tomography

# %%
import numpy as np
import itertools
from functools import reduce

# Set printing options for clear terminal output
np.set_printoptions(precision=4, suppress=True, linewidth=120)


# %% [markdown]
#  ## 1. Create the Orthonormal Basis
# 
#  Choose `n_qubits`. Create a tensor of self-adjoint, orthonormal matrices to use as a basis for the state matrix.
# 
#  A $(2^n)^2$ vector of $2^n \times 2^n$ matrices is generated.
# 
#  We build each matrix as the tensor product of the combinations of the 1-qubit basis matrices.

# %%
n_qubits = 2 # CAUTION: n_qubits > 6 crashes due to memory limits
dim = 2**n_qubits
n_params = dim**2

# Define 1-qubit Pauli matrices (unscaled)
identity = np.array([[1, 0], [0, 1]], dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

base_1q = np.array([identity, pauli_x, pauli_y, pauli_z])

# Generate combinations for n qubits
combinations = itertools.product(base_1q, repeat=n_qubits)

# Compute tensor products
matrix_base = np.array([reduce(np.kron, combo) for combo in combinations])

# Apply global scaling factor to ensure orthonormality (Tr(Gamma_mu Gamma_nu) = delta_mu,nu)
scale_factor = 2**(-n_qubits / 2)
matrix_base = matrix_base * scale_factor


# %%
# Test: check shape and orthonormality
print("--- Basis Generation ---")
print(f"Number of qubits: {n_qubits}")
print(f"Number of parameters (4^n): {n_params}")
print("Basis tensor shape:", matrix_base.shape)

# Check orthonormality of a random element (should print 1.0 + 0.0j)
ortho_check = np.trace(matrix_base[1] @ matrix_base[1].conj().T)
print(f"Orthonormality check (one element): {ortho_check}")


# %% [markdown]
#  ## 2. Generate the True State Matrix
# 
#  We generate a valid density matrix using the Ginibre ensemble approach to ensure it is
# 
#  positive semi-definite (rho >= 0) and has unit trace (Tr(rho) = 1).

# %%
def generate_random_state():
    # Generate a random positive semi-definite matrix
    G = np.random.normal(0, 1, (dim, dim)) + 1j * np.random.normal(0, 1, (dim, dim))
    rho_unnormalized = G @ G.conj().T

    # Enforce Tr(rho) = 1
    rho = rho_unnormalized / np.trace(rho_unnormalized)
    
    return rho

# %%
# Test: check physical constraints
rho_test = generate_random_state()
print("\n--- Generated State ---")
print("Trace(rho_test) [Should be 1.0]:", np.round(np.real(np.trace(rho_test)), 6))
print("Trace(rho_test^2) [Should be <= 1.0]:", np.round(np.real(np.trace(rho_test @ rho_test)), 6))

eigenvalues = np.sort(np.linalg.eigvalsh(rho_test))
print("Eigenvalues (must be >= 0):")
print(np.round(eigenvalues, 4))


# %% [markdown]
#  ## 3. Define the POVM
# 
#  The POVM has $4^n$ elements and must be informationally complete (IC).
# 
#  For a single qubit, we use 4 projectors mapping to an IC-POVM.
# 
#  The $n$-qubit POVM is the tensor product of all combinations.

# %%
# 1-qubit kets (defined as 1D arrays, np.outer will handle the matrices correctly)
ket_z_plus = np.array([1, 0], dtype=complex)
ket_z_minus = np.array([0, 1], dtype=complex)
ket_x_plus = (ket_z_plus + ket_z_minus) / np.sqrt(2)
ket_x_minus = (ket_z_plus - ket_z_minus) / np.sqrt(2)
ket_y_plus = (ket_z_plus + 1j * ket_z_minus) / np.sqrt(2)
ket_y_minus = (ket_z_plus - 1j * ket_z_minus) / np.sqrt(2)

# Select 4 independent kets to form the minimal IC-POVM
kets_1q = np.array([ket_z_plus, ket_z_minus, ket_x_plus, ket_y_minus])

# Build the projectors P = |psi><psi|
povm_1q = np.array([np.outer(k, k.conj()) for k in kets_1q])

# Generate n-qubit POVM via tensor products
combinations_povm = itertools.product(povm_1q, repeat=n_qubits)
povm = np.array([reduce(np.kron, combo) for combo in combinations_povm])

# %%
# Test
print("\n--- POVM ---")
print("POVM shape: ", povm.shape)


# %% [markdown]
#  ## 4. Probability Vector
# 
#  Vector $p$ containing the expected probabilities for each POVM outcome.

# %%
# Computes p_mu = Tr[P_mu * rho] for all projectors simultaneously

def probability_vector(rho, povm):
    return np.real(np.einsum('mij,ji->m', povm, rho))

# %%
# Test
p_vec_test = probability_vector(rho_test)
print(p_vec_test)


# %% [markdown]
#  ## 5. Linear Inversion
# 
#  Building the design matrix $B$ and the reconstruction operator $M$.

# %%

B = np.real(np.einsum('mij,nji->mn', povm, matrix_base))
B_inv = np.linalg.inv(B)

M = np.einsum('nij,nm->mij', matrix_base, B_inv)

def linear_inversion(frequency_vector):
    rho_reconstructed = np.tensordot(frequency_vector, M, axes=1)
    return rho_reconstructed


# %%
# Test Reconstruction
rho_test_reconstructed= linear_inversion(rho_test, povm)
print("\n--- Linear Inversion ---")
print("B matrix shape:", B.shape)
print("M operator shape:", M.shape)

print("rho_test real:\n", rho_test)
print("rho_test reconstructed:\n", rho_test_reconstructed)


# %% [markdown]
#  ## 6. Evaluation Metrics
# 
#  - **Trace Distance**: $T(\rho, \sigma) = \frac{1}{2} ||\rho - \sigma||_1$
# 
#  - **Fidelity**: $F(\rho, \sigma) = (\text{Tr}[\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}])^2$

# %%
def trace_distance(rho_1, rho_2):
    diff = rho_1 - rho_2
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))

def sqrt_hermitian_matrix(matrix):
    eigenvalues, U = np.linalg.eigh(matrix)
    sqrt_eigenvalues = np.sqrt(eigenvalues) # np.clip(eigenvalues, 0, None) maybe?
    return U @ np.diag(sqrt_eigenvalues) @ U.conj().T

def fidelity(rho_1, rho_2):
    rho_1_sqrt = sqrt_hermitian_matrix(rho_1)
    inner_matrix = rho_1_sqrt @ rho_2 @ rho_1_sqrt
    inner_sqrt = sqrt_hermitian_matrix(inner_matrix)
    
    # Real part handles microscopic imaginary floating-point artifacts
    return np.real(np.trace(inner_sqrt))**2


# %%
# TEST METRICS
print("Trace Distance (rho_test, rho_test):", np.round(trace_distance(rho_test, rho_test), 6))
print("Fidelity (rho_test, rho_test):", np.round(fidelity(rho_test, rho_test), 6))

print("\nTrace Distance (Reconstruction):", np.round(trace_distance(rho_test, rho_test_reconstructed), 10))
print("Fidelity (Reconstruction):", np.round(fidelity(rho_test, rho_test_reconstructed), 10))



# %% [markdown]
# # Simulation, 1 Qubit
# 
# Create `n_states` random states.
# Measure each of them `n_measures` times, each projector of the POVM is measured `n_measures_proj` times
# 

# %%
n_states = 10
n_measures_proj = 10
n_measures = n_measures_proj * n_params
print('n_measures: ',n_measures)

trace_distances = []
fidelities = []

for _ in range(n_states):
    state = generate_random_state()

    prob_vec = probability_vector(state, povm)
    
    counts = np.random.binomial(n_measures_proj, prob_vec)
    freq_vec = counts / n_measures_proj
    
    reconstructed_state = linear_inversion(freq_vec)
    
    td = trace_distance(state, reconstructed_state)
    fid = fidelity(state, reconstructed_state)
    
    trace_distances.append(td)
    fidelities.append(fid)
    
    print("\n--- Reconstruction Accuracy ---")
    print("Trace Distance (Reconstruction):", np.round(td, 10))
    print("Fidelity (Reconstruction):", np.round(fid, 10))

print(f"\nAverage Trace Distance: {np.mean(trace_distances):.4f} ± {np.std(trace_distances):.4f}")
print(f"\nAverage Fidelity: {np.mean(fidelities):.4f} ± {np.std(fidelities):.4f}")


