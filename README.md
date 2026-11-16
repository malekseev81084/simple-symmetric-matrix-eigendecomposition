# simple-symmetric-matrix-eigendecomposition

A small, dependency-free Python library that diagonalizes real symmetric matrices using the Jacobi eigenvalue algorithm.

## Usage

```python
from simple_symmetric_matrix_eigendecomposition import jacobi_eigen, reconstruct

A = [[2.0, 1.0], [1.0, 2.0]]
eigenvalues, eigenvectors = jacobi_eigen(A)
# eigenvalues == [1.0, 3.0] (ascending)
# eigenvectors[i] is the i-th eigenvector

B = reconstruct(eigenvalues, eigenvectors)
# B is approximately A
```

The module also exports `is_symmetric(matrix)` and the exception `SymmetricEigendecompositionError`.

## Why this exists

The Jacobi eigenvalue algorithm is short, stable for symmetric matrices, and needs only the standard library. The trade-off is speed: it is slower than tridiagonal QR for large matrices, but this library targets small matrices and cases where pulling in NumPy is not desirable. It also makes the algorithm easy to read and audit.

## Edge cases

The input must be a non-empty square matrix that is symmetric within a small tolerance. If it is not, `jacobi_eigen` raises `SymmetricEigendecompositionError`. Empty matrices are rejected rather than treated as zero-dimensional. The implementation uses a cyclic sweep and stops when the sum of squared off-diagonal entries falls below `tol * tol`; the default tolerance is `1e-12`.

Eigenvalues are sorted in ascending order. Eigenvectors are returned as a list of lists, where `eigenvectors[i][j]` is the j-th component of the i-th eigenvector. The sign of each eigenvector is arbitrary; tests that need a fixed sign should normalize it themselves.
