"""
Derived module from dmdbase.py for higher order dmd.

Reference:
- S. L Clainche, J. M. Vega, Higher Order Dynamic Mode Decomposition.
Journal on Applied Dynamical Systems, 16(2), 882-925, 2017.
"""
from past.utils import old_div
import numpy as np
import scipy as sp
from scipy.linalg import pinv2
from mosessvd import MOSESSVD
from numba import jit

from .mosesdmdbase import MOSESDMDBase


def pinv(x): return pinv2(x, rcond=10 * np.finfo(float).eps)


class MOSESDMD(MOSESDMDBase):
    """
    MOSES SVD based Higher Order Dynamic Mode Decomposition

    :param int svd_rank: rank truncation in SVD. If 0, the method computes the
        optimal rank and uses it for truncation; if positive number, the method
        uses the argument for the truncation; if -1, the method does not
        compute truncation.
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    :param int d: the new order for spatial dimension of the input snapshots.
        Default is 1.
    :param int chunk_size: the horizontal size for the chunks given to MOSES SVD.
    :param numpy.dtype dtype: The desired datatype used for calculations.
    (might be removed in the future)
    :param boolean projection: Whether to use V or the projection of U for
    DMD. The second option is better, but requires more computations.
    Default is True.
    :param int or tring sqrt_K: Choose the method to calculate K. Default is True.
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False, d=1,
                 chunk_size=None, dtype=np.complex64, projection=True,
                 sqrt_K=True):
        super(MOSESDMD, self).__init__(svd_rank, tlsq_rank, exact, opt)
        self.d = d
        self.chunk_size = chunk_size
        self.U = None
        self.s = None
        self.V = None
        self.K_list = None
        self.M = None
        self.dtype = dtype
        self.projection = projection
        self.sqrt_K = sqrt_K
        self.K_eigvec = None

    def linsolve(self, A, B):
        return np.matmul(B, np.linalg.inv(A))

    # @profile
    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        if X.dtype != self.dtype:
            X = X.astype(self.dtype)

        self._snapshots = X

        n_samples = self._snapshots.shape[1]

        # X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)   not implemented

        msvd = MOSESSVD(rank=self.svd_rank)

        # MOSES SVD iteration loop
        i = -1
        for i in range(self.d-1, self._snapshots.shape[1] - self.chunk_size, self.chunk_size):
            chunk = [self._snapshots[:, i-j:i-j+self.chunk_size] for j in range(self.d)]
            chunk = np.vstack(chunk)
            msvd.update(chunk)

        # final chunk that contains the remaining snapshots
        chunk = [self._snapshots[:, i+1-j+self.chunk_size: self._snapshots.shape[1]-j] for j in range(self.d)]
        chunk = np.vstack(chunk)
        msvd.update(chunk)

        # get the SVD matrices
        U, s, V = msvd.S.astype(self.dtype), msvd.Gamma.astype(self.dtype), msvd.Q.astype(self.dtype)
        self.U, self.s, self.V = U, s, V

        M = np.zeros((self.svd_rank, self._snapshots.shape[1] - self.d)).astype(self.dtype)
        U_conj = np.ascontiguousarray(U.conj().T)

        # calculate M
        if self.projection:
            for i in range(self.svd_rank):
                M[i, :] = self.M_projection_value(self._snapshots, U_conj, i, self.d, self._snapshots.shape[1] - self.d,
                                                  self.dtype)
        else:
            M = s.dot(V.conj().T)

        self.M = M

        # get the time shifted MX and MY
        MX = M[:, :-1]
        MY = M[:, 1:]

        # calculate the forward and backward operators
        Kf = MY.dot(pinv(MX))
        Kb = MX.dot(pinv(MY))
        Kbinv = pinv(Kb)
        if self.sqrt_K == "mean":
            K = (Kf + Kbinv) / 2
        elif self.sqrt_K:
            K = sp.linalg.sqrtm(Kf.dot(Kbinv))
        else:
            K = Kf
        self.Atilde = K
        K_eigval, K_eigvec = np.linalg.eig(K)
        self._eigs = K_eigval
        self.K_eigvec = K_eigvec

        # calculate the modes truncated to the original size
        self._modes = U[:self._snapshots.shape[0]].dot(K_eigvec.astype(self.dtype))

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        return self

    def _compute_amplitudes(self, modes, snapshots, eigs, opt):
        """
        Compute the amplitude coefficients. If `opt` is False the amplitudes
        are computed by minimizing the error between the modes and the first
        snapshot; if `opt` is True the amplitudes are computed by minimizing
        the error between the modes and all the snapshots, at the expense of
        bigger computational cost.
        :param numpy.ndarray modes: 2D matrix that contains the modes, stored
            by column.
        :param numpy.ndarray snapshots: 2D matrix that contains the original
            snapshots, stored by column.
        :param numpy.ndarray eigs: array that contains the eigenvalues of the
            linear operator.
        :param bool opt: flag for computing the optimal amplitudes of the DMD
            modes, minimizing the error between the time evolution and all
            the original snapshots. If false the amplitudes are computed
            using only the initial condition, that is snapshots[0].
        :return: the amplitudes array
        :rtype: numpy.ndarray
        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        """
        if opt:
            # compute the vandermonde matrix
            omega = old_div(np.log(eigs), self.original_time['dt'])
            vander = np.exp(
                np.multiply(*np.meshgrid(omega, self.dmd_timesteps))).T

            # perform svd on all the snapshots
            # msvd = MOSESSVD(rank=self.svd_rank)
            # U, s, V = msvd.iterated_svd(snapshots, b=self.svd_rank+1)
            # V = V.conj().T
            # U, s, V = np.linalg.svd(self._snapshots, full_matrices=False)
            U, s, M = self.U, self.s, self.M
            K_eigvec = self.K_eigvec
            sinv = np.diag(np.reciprocal(np.diag(s)))
            V = np.dot(sinv, M).conj().T

            vander = vander[:,vander.shape[1] - V.shape[0]:]

            P = np.multiply(
                np.dot(K_eigvec.conj().T, K_eigvec),
                np.conj(np.dot(vander, vander.conj().T)))
            tmp = np.dot(V, s.conj().T)
            q = np.conj(np.diag(np.dot(np.dot(vander, tmp), K_eigvec)))

            # b optimal
            a = np.linalg.solve(P, q)
        else:
            a = np.linalg.lstsq(modes, snapshots.T[0], rcond=None)[0]

        return a

    @staticmethod
    @jit(nopython=True)
    def M_projection_value(snapshots, S_conj, index_i, d, length_j, dtype):
        """
        Generates the i-th row from the matrix product of U and the stacked snapshots.
        This projects the stacked snapshots to the subspace of U
        Parameters
        ----------
        snapshots : numpy.ndarray
            Snapshot matrix
        U_conj : numpy.ndarray
            Complex conjugate of U matrix. It is more efficient to do the
            conjugate transpose outside this method
        index_i : int
            Index i for the M matrix
        d : int
            stacking depth of the snapshots
        dtype : numpy.dtype
            Target datatype.

        Returns
        -------
        value_row : The i-th row of M

        """
        S_row = S_conj[index_i]
        snapshot_length = snapshots.shape[0]
        value_row = np.zeros(length_j).astype(dtype)
        for index_j in range(length_j):
            value = dtype(0)
            for m_slice_nr in range(d):
                m_slice = snapshots[:, index_j+d-1 - m_slice_nr]
                s_slice = S_row[m_slice_nr * snapshot_length : (m_slice_nr+1) * snapshot_length]
                value += s_slice.dot(m_slice)
            value_row[index_j] = value
        return value_row
