import math
import unittest

from simple_symmetric_matrix_eigendecomposition import (
    SymmetricEigendecompositionError,
    is_symmetric,
    jacobi_eigen,
    reconstruct,
)


def assert_close(testcase, got, want, tol=1e-8):
    testcase.assertLessEqual(abs(got - want), tol, msg=f"got {got}, want {want}")


class TestIsSymmetric(unittest.TestCase):
    def test_square_symmetric(self):
        self.assertTrue(is_symmetric([[1.0, 2.0], [2.0, 3.0]]))

    def test_non_square(self):
        self.assertFalse(is_symmetric([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_asymmetric(self):
        self.assertFalse(is_symmetric([[1.0, 2.0], [3.0, 4.0]]))

    def test_empty(self):
        self.assertTrue(is_symmetric([]))

    def test_tolerance(self):
        self.assertTrue(is_symmetric([[1.0, 1.0 + 1e-13], [1.0, 2.0]]))
        self.assertFalse(is_symmetric([[1.0, 1.0 + 1e-9], [1.0, 2.0]]))


class TestJacobiEigen(unittest.TestCase):
    def test_diagonal_matrix(self):
        matrix = [[3.0, 0.0], [0.0, 1.0]]
        vals, vecs = jacobi_eigen(matrix)
        assert_close(self, vals[0], 1.0)
        assert_close(self, vals[1], 3.0)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 2)

    def test_known_2x2(self):
        matrix = [[2.0, 1.0], [1.0, 2.0]]
        vals, vecs = jacobi_eigen(matrix)
        assert_close(self, vals[0], 1.0)
        assert_close(self, vals[1], 3.0)
        # Eigenvector for lambda=1 is (1, -1)/sqrt(2).
        v0 = vecs[0]
        assert_close(self, abs(v0[0]), 1.0 / math.sqrt(2.0))
        assert_close(self, abs(v0[1]), 1.0 / math.sqrt(2.0))
        self.assertLess(v0[0] * v0[1], 0.0)

    def test_identity(self):
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        vals, vecs = jacobi_eigen(matrix)
        for val in vals:
            assert_close(self, val, 1.0)
        for vec in vecs:
            norm = math.sqrt(sum(x * x for x in vec))
            assert_close(self, norm, 1.0)

    def test_diagonal_3x3(self):
        matrix = [[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 9.0]]
        vals, _ = jacobi_eigen(matrix)
        assert_close(self, vals[0], 2.0)
        assert_close(self, vals[1], 5.0)
        assert_close(self, vals[2], 9.0)

    def test_nearly_diagonal(self):
        matrix = [[1.0, 1e-10], [1e-10, 1.0]]
        vals, _ = jacobi_eigen(matrix)
        assert_close(self, vals[0], 1.0)
        assert_close(self, vals[1], 1.0)

    def test_non_symmetric_raises(self):
        with self.assertRaises(SymmetricEigendecompositionError):
            jacobi_eigen([[1.0, 2.0], [3.0, 4.0]])

    def test_non_square_raises(self):
        with self.assertRaises(SymmetricEigendecompositionError):
            jacobi_eigen([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_empty_raises(self):
        with self.assertRaises(SymmetricEigendecompositionError):
            jacobi_eigen([])

    def test_single_element(self):
        vals, vecs = jacobi_eigen([[7.0]])
        assert_close(self, vals[0], 7.0)
        assert_close(self, vecs[0][0], 1.0)


class TestReconstruct(unittest.TestCase):
    def test_reconstruct_2x2(self):
        matrix = [[2.0, 1.0], [1.0, 2.0]]
        vals, vecs = jacobi_eigen(matrix)
        rebuilt = reconstruct(vals, vecs)
        for i in range(2):
            for j in range(2):
                assert_close(self, rebuilt[i][j], matrix[i][j])

    def test_reconstruct_3x3(self):
        matrix = [
            [4.0, 1.0, 2.0],
            [1.0, 3.0, 0.5],
            [2.0, 0.5, 5.0],
        ]
        vals, vecs = jacobi_eigen(matrix)
        rebuilt = reconstruct(vals, vecs)
        for i in range(3):
            for j in range(3):
                assert_close(self, rebuilt[i][j], matrix[i][j], tol=1e-7)

    def test_reconstruct_mismatched_lengths_raises(self):
        with self.assertRaises(SymmetricEigendecompositionError):
            reconstruct([1.0, 2.0], [[1.0], [0.0]])

    def test_reconstruct_wrong_eigenvector_length_raises(self):
        with self.assertRaises(SymmetricEigendecompositionError):
            reconstruct([1.0], [[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
