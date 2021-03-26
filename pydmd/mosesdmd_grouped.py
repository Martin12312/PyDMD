"""
Derived module from dmdbase.py for higher order dmd.

Reference:
- S. L Clainche, J. M. Vega, Higher Order Dynamic Mode Decomposition.
Journal on Applied Dynamical Systems, 16(2), 882-925, 2017.
"""
import numpy as np
import scipy as sp
from scipy.linalg import pinv2
from mosessvd import MOSESSVD
from numba import jit
from past.utils import old_div

from .mosesdmdbase import MOSESDMDBase


def pinv(x): return pinv2(x, rcond=10 * np.finfo(float).eps)


class MOSESDMD_grouped(MOSESDMDBase):
    """
    MOSESDMD for processing multiple groups of sequential snapshots.
    Input is a list of all the groups of snapshots
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False, d=1,
                 chunk_size=None, dtype=np.complex64, projection=True,
                 sqrt_K=True, compute_amplitudes_method=0):
        super(MOSESDMD_grouped, self).__init__(svd_rank, tlsq_rank, exact, opt)
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
        self.compute_amplitudes_method = compute_amplitudes_method

    def linsolve(self, A, B):
        return np.matmul(B, np.linalg.inv(A))

    # @profile
    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        for i in range(len(X)):
            if X[i].dtype != self.dtype:
                X[i] = X[i].astype(self.dtype)

        # convert the input list to a tuple
        # necessary for numba
        X = tuple(X)
        self._snapshots = X

        # X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)   not implemented

        msvd = MOSESSVD(rank=self.svd_rank)

        # calculate the width of M
        M_width = 0
        for group in self._snapshots:
            M_width += group.shape[1] - self.d + 1

        # number of full chunks that fit in the stacked snapshots.
        whole_chunks = int(np.floor(M_width / self.chunk_size)) - 1

        # MOSES SVD iteration loop
        i = -1
        for i in range(whole_chunks):
            chunk = self.get_chunk(self.snapshots, self.chunk_size, i, self.d, False)
            msvd.update(chunk)
        # final chunk that contains the remaining snapshots
        chunk = self.get_chunk(self.snapshots, self.chunk_size, i+1, self.d, True)
        msvd.update(chunk)

        # get the SVD matrices
        U, s, V = msvd.S.astype(self.dtype), msvd.Gamma.astype(self.dtype), msvd.Q.astype(self.dtype)
        self.U, self.s, self.V = U, s, V

        M = np.zeros((self.svd_rank, M_width)).astype(self.dtype)
        U_conj = np.ascontiguousarray(U.conj().T)   # for M_projection_value()

        # calculate M
        if self.projection:
            # loop that projects the stacked snapshots onto U
            for i in range(self.svd_rank):
                M[i, :] = self.M_projection_value(self._snapshots, U_conj, i, self.d, self.dtype)
        else:
            M = s.dot(V.conj().T)

        self.M = M

        # calculates which collumns to delete to get MX and MY
        # for MX the first collumn for each time series is deleted
        # for MY the last collumn fr each time series is deleted
        len_snaps_each = np.array([group.shape[1] - self.d + 1 for group in self._snapshots])
        ind_snaps_groups = np.array([0])
        ind_snaps_groups = np.append(ind_snaps_groups, np.cumsum(len_snaps_each))

        ind_del_0 = ind_snaps_groups[:-1]
        ind_del_1 = ind_snaps_groups[1:] - 1

        MX = np.delete(M, ind_del_1, axis=1)
        MY = np.delete(M, ind_del_0, axis=1)

        Kf = MY.dot(pinv(MX))
        Kb = MX.dot(pinv(MY))
        Kbinv = pinv(Kb)
        # How to calculate K from Kb and Kf
        if self.sqrt_K == "mean":
            K = (Kf + Kbinv) / 2
        if self.sqrt_K == "back":
            K = Kbinv
        elif self.sqrt_K:
            K = sp.linalg.sqrtm(Kf.dot(Kbinv))
        else:
            K = Kf
        self.Atilde = K
        K_eigval, K_eigvec = np.linalg.eig(K)
        self._eigs = K_eigval

        modes_full = U.dot(K_eigvec.astype(self.dtype))
        self._modes = modes_full[:self._snapshots[0].shape[0]]

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': self._snapshots[0].shape[1] - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': self._snapshots[0].shape[1] - 1, 'dt': 1}

        """
        Determines the amplitude computation method
        0: Globally fitted amplitudes using U S V, but truncated to the length of a single snapshot
        1: Globally fitted amplitudes using full U S V, can cause memory crash
        2: Amplitudes of the first time series
        """

        if self.compute_amplitudes_method == 0:
            self._b = self._compute_amplitudes_average(modes_full, self._snapshots,
                                                self._eigs, self.opt, method=0)
        if self.compute_amplitudes_method == 1:
            self._b = self._compute_amplitudes_average(modes_full, self._snapshots,
                                                self._eigs, self.opt, method=1)
        if self.compute_amplitudes_method == 2:
            self._b = self._compute_amplitudes(self._modes, self._snapshots[0],
                                                self._eigs, self.opt)

        return self

    # get the i-th reconstructed time series
    def reconstructed_data_i(self, i):
        return self.modes.dot(self.dynamics_i(i))

    # get the dynamics of the i-th reconstructed time series
    def dynamics_i(self, i):
        self.original_time = {'t0': 0, 'tend': self._snapshots[i].shape[1] - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': self._snapshots[i].shape[1] - 1, 'dt': 1}

        amplitudes = self._compute_amplitudes(self._modes, self._snapshots[i],
                                              self._eigs, self.opt)

        omega = old_div(np.log(self.eigs), self.original_time['dt'])
        vander = np.exp(
            np.outer(omega, self.dmd_timesteps - self.original_time['t0']))
        return vander * amplitudes[:, None]

    # compute the globally fitted initial amplitudes
    def _compute_amplitudes_average(self, modes, snapshots, eigs, opt, method):
        """
        Compute the amplitude coefficients for each trajectory. If `opt` is False the amplitudes
        are computed by minimizing the error between the modes and the first
        snapshot; if `opt` is True the amplitudes are computed by minimizing
        the error between the modes and all the snapshots, at the expense of
        bigger computational cost.
        :param numpy.ndarray modes: 2D matrix that contains the modes, stored
            by column.
        :param numpy.ndarray snapshots: 2D matrix that contains the original
            snapshots, stored by column.
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
            timesteps = []
            dt = self.original_time['dt']
            for sn in snapshots:
                t0 = 0
                t1 = dt * (sn.shape[1] - 1)
                timesteps.append(np.arange(t0, t1+1e-10, dt))

            timesteps = np.hstack(timesteps)

            # use the first n rows for the computation
            # using the full matrices is very expensive
            n = self._snapshots[0].shape[0]
            if method == 1:
                U, s, V = self.U[:n], np.diag(self.s)[:n], self.V.conj().T
                modes = modes[:n]
            else:
                U, s, V = self.U, np.diag(self.s), self.V.conj().T
            timesteps = timesteps[:V.shape[1]]

            omega = old_div(np.log(eigs), dt)
            vander = np.exp(
                np.multiply(*np.meshgrid(omega, timesteps))).T

            P = np.multiply(
                np.dot(modes.conj().T, modes),
                np.conj(np.dot(vander, vander.conj().T)))
            tmp = (np.dot(np.dot(U, np.diag(s)), V)).conj().T
            q = np.conj(np.diag(np.dot(np.dot(vander, tmp), modes)))

            a = np.linalg.solve(P, q)
        else:
            a = np.linalg.lstsq(modes, snapshots.T[0], rcond=None)[0]

        return a

    @staticmethod
    @jit(nopython=True)
    def M_projection_value(snapshots, U_conj, index_i, d, dtype):
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
        U_row = U_conj[index_i]
        snapshot_length = snapshots[0].shape[0]
        length_j = 0
        for i in snapshots:
            length_j += i.shape[1] - d + 1
        value_row = np.zeros(length_j).astype(dtype)
        index_j = 0
        for group_ind in range(len(snapshots)):
            group = snapshots[group_ind]
            for group_j in range(group.shape[1] - d + 1):
                value = dtype(0)
                for m_slice_nr in range(d):
                    m_slice = group[:, group_j+d-1 - m_slice_nr]
                    u_slice = U_row[m_slice_nr * snapshot_length : (m_slice_nr+1) * snapshot_length]
                    value += u_slice.dot(m_slice)
                value_row[index_j] = value
                index_j += 1
        return value_row

    # This is a very ugly method. Be warned
    @staticmethod
    def get_chunk(snapshots, chunk_size, i, d, get_remaining):
        """
        This method generates stacked snapshot data chunks for the MOSES SVD loop.
        It handles the stacking of the snapshots. It ensures proper handling of
        both eds of a time series. It ensures that data from one time series
        doesn't bleed over to an adjacent one at both ends due to the time-delay stacking.
        The last chunk will usually be between chunk_size and 2*chunk_size in size.
        This is done to avoid generating a small final chunk as it can break MOSES SVD.
        Parameters
        ----------
        snapshots : numpy.ndarray
            Snapshot matrix
        chunk_size : int
            Desired size of a chunk.
        i : int
            Index of the chunk.
        d : int
            stacking depth of the snapshots
        get_remaining : boolean
            If set to True, generates a chunk containing all the ramaining data.
            Intended for the final chunk.

        Returns
        -------
        chunk : numpy.ndarray
            A chunk of stacked snapshot data for MOSES SVD.

        """

        """
        The way this works is by generating arrays that label each snapshot by
        the index of the source time series (group_numbers) and
        a modified index that excludes the first d-1 snapshots from each series (effective_indexes).
        The start and end j indexes in the stacked snapshot matrix is calculated.
        numpy.nonzero is then used to determine the right snapshot indexes to start and end at.
        Then the used time series indexes are determined.
        The used time series are then looped over and the snapshots are stacked.

        It is important to note that the stacked snapshot matrix is not explicitly
        defined, but generated on the fly by slicing and stacking.

        Time series are labeled as groups in the code
        """

        total_width = 0
        for s in snapshots:
            total_width += s.shape[1] - d + 1

        # Indexes for the start and end collumns in the stacked snapshot matrix
        chunk_start = chunk_size * i
        chunk_end = chunk_size * (i + 1)

        j_index = 0
        effective_indexes = []
        group_numbers = []
        group_lengths_cumulative = [0]
        group_length_counter = 0
        # generate label arrays later used for index matching
        for i, s in enumerate(snapshots):
            row_length = s.shape[1]
            row_indexes = np.arange(row_length) - d + 1
            numbers = np.ones(row_length) * i
            row_indexes += j_index
            j_index += row_length - d + 1
            row_indexes[:d-1] = -1
            effective_indexes = effective_indexes + list(row_indexes)
            group_numbers = group_numbers + list(numbers)
            group_length_counter += s.shape[1]
            group_lengths_cumulative.append(group_length_counter)

        effective_indexes = np.array(effective_indexes).astype(np.int)
        group_numbers = np.array(group_numbers).astype(np.int)
        group_lengths_cumulative = np.array(group_lengths_cumulative).astype(np.int)

        true_indexes = np.arange(len(effective_indexes))
        start_index = np.nonzero((effective_indexes - chunk_start) == 0)[0][0]
        end_index = np.nonzero((effective_indexes - chunk_end) == 0)[0]

        # checks wether to include all of the remaining snapshots
        if list(end_index) == [] or get_remaining:
            end_index = true_indexes[-1]
        else:
            end_index = end_index[0]

        group_start = group_numbers[start_index]
        group_end = group_numbers[end_index]

        chunk = []

        to_merge = []
        for group_ind in range(group_start, group_end+1):   #this loops over all the groups that the chunk takes from
            group = snapshots[group_ind]
            group_start_index = start_index - group_lengths_cumulative[group_ind]
            group_end_index = end_index - group_lengths_cumulative[group_ind]

            if group_end_index > group.shape[1]:    #end is not in this group
                if group_start_index <= 0:          #start is in the previous group
                    to_add = [group[:, d-1-j: -j + group.shape[1]] for j in range(d)]
                    to_add = np.vstack(to_add)
                    to_merge.append(to_add)
                else:                               #start is in this group
                    to_add = [group[:, -j+group_start_index: -j + group.shape[1]] for j in range(d)]
                    to_add = np.vstack(to_add)
                    to_merge.append(to_add)
            else:                                   #end is in this group
                if group_start_index < 0:           #start is in the previous group
                    to_add = [group[:, d-1-j: -j + group_end_index] for j in range(d)]
                    to_add = np.vstack(to_add)
                    to_merge.append(to_add)
                else:                               #start is in this group
                    to_add = [group[:, -j+group_start_index: -j + group_end_index] for j in range(d)]
                    to_add = np.vstack(to_add)
                    to_merge.append(to_add)

        chunk = np.hstack(to_merge)
        return chunk