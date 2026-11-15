import math


class SymmetricEigendecompositionError(ValueError):
    """Raised when the input is not a valid real symmetric matrix."""


def is_symmetric(matrix, tol=1e-12):
    """Return True if *matrix* is a square 2D sequence that is symmetric within *tol*.

    The check is exact for the diagonal and tolerance-based off-diagonal.
    """
    n = len(matrix)
    if n == 0:
        return True
    for row in matrix:
        if len(row) != n:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > tol:
                return False
    return True


def _validate(matrix):
    n = len(matrix)
    if n == 0:
        raise SymmetricEigendecompositionError("matrix must be non-empty")
    for row in matrix:
        if len(row) != n:
            raise SymmetricEigendecompositionError("matrix must be square")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-10:
                raise SymmetricEigendecompositionError(
                    "matrix must be symmetric within tolerance"
                )


def jacobi_eigen(matrix, max_sweeps=100, tol=1e-12):
    """Diagonalize a real symmetric matrix with the cyclic Jacobi method.

    Returns ``(eigenvalues, eigenvectors)`` where *eigenvalues* is a list of
    floats and *eigenvectors* is a list of lists: ``eigenvectors[i][j]`` is the
    j-th component of the i-th eigenvector. Eigenvalues are sorted in ascending
    order. The matrix must be square and symmetric within a small tolerance;
    otherwise :class:`SymmetricEigendecompositionError` is raised.

    The Jacobi method is chosen because it is short, numerically stable for
    symmetric matrices, and needs no external linear algebra library. It is
    slower than tridiagonal QR for large matrices, but this library targets
    small matrices and educational use.
    """
    _validate(matrix)
    n = len(matrix)
    a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    # V starts as the identity; columns accumulate the eigenvectors.
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += a[p][q] * a[p][q]
        if off <= tol * tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= tol:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                # Stable root selection: avoid cancellation when theta is large.
                if theta >= 0.0:
                    t = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
                else:
                    t = -1.0 / (-theta + math.sqrt(1.0 + theta * theta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                app = a[p][p]
                aqq = a[q][q]
                apq = a[p][q]
                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0

                for i in range(n):
                    if i != p and i != q:
                        aip = a[i][p]
                        aiq = a[i][q]
                        a[i][p] = c * aip - s * aiq
                        a[p][i] = a[i][p]
                        a[i][q] = s * aip + c * aiq
                        a[q][i] = a[i][q]

                for i in range(n):
                    vip = v[i][p]
                    viq = v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq

    eigenvalues = [a[i][i] for i in range(n)]
    # Sort ascending; carry the corresponding eigenvector columns.
    order = sorted(range(n), key=lambda i: eigenvalues[i])
    eigenvalues = [eigenvalues[i] for i in order]
    eigenvectors = []
    for i in order:
        eigenvectors.append([v[row][i] for row in range(n)])
    return eigenvalues, eigenvectors


def reconstruct(eigenvalues, eigenvectors):
    """Rebuild the matrix from its eigenvalues and eigenvectors.

    ``eigenvectors`` is the list-of-lists returned by :func:`jacobi_eigen`,
    where ``eigenvectors[i]`` is the i-th eigenvector. The result is
    ``sum_i lambda_i * v_i * v_i^T``. This is useful for verifying that the
    decomposition is accurate without relying on matrix multiplication.
    """
    n = len(eigenvalues)
    if n != len(eigenvectors):
        raise SymmetricEigendecompositionError(
            "eigenvalues and eigenvectors must have the same length"
        )
    result = [[0.0] * n for _ in range(n)]
    for k in range(n):
        lam = eigenvalues[k]
        vec = eigenvectors[k]
        if len(vec) != n:
            raise SymmetricEigendecompositionError("eigenvector has wrong length")
        for i in range(n):
            vi = vec[i]
            for j in range(n):
                result[i][j] += lam * vi * vec[j]
    return result
