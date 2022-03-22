# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Common utility functions for tomography fitters.
"""

from typing import Optional, Tuple, Callable, Sequence
import functools
import numpy as np

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
)


def lstsq_data(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int]] = None,
    preparation_qubits: Optional[Tuple[int]] = None,
    conditional_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked vectorized basis matrix A for least squares."""
    if measurement_basis is None and preparation_basis is None:
        raise AnalysisError("`measurement_basis` and `preparation_basis` cannot both be None")

    # Get leading dimension of returned matrix
    size = outcome_data.size
    mdim = 1
    pdim = 1
    cdim = 1
    num_cond = 0

    # Get full and conditional measurement basis dimensions
    if measurement_basis:
        bsize, num_meas = measurement_data.shape
        if not measurement_qubits:
            measurement_qubits = tuple(range(num_meas))

        # Partition measurement qubits into conditional measurement qubits and
        # regular measurement qubits
        if conditional_indices is not None:
            conditional_indices = tuple(conditional_indices)
            conditional_qubits = tuple(measurement_qubits[i] for i in conditional_indices)
            measurement_qubits = tuple(
                qubit for i, qubit in enumerate(measurement_qubits) if i not in conditional_indices
            )
            num_cond = len(conditional_indices)
            cdim = np.prod(measurement_basis.outcome_shape(conditional_qubits))

        mdim = np.prod(measurement_basis.matrix_shape(measurement_qubits))

    # Get preparation basis dimensions
    if preparation_basis:
        bsize, num_prep = preparation_data.shape
        if not preparation_qubits:
            preparation_qubits = tuple(range(num_prep))
        pdim = np.prod(preparation_basis.matrix_shape(preparation_qubits))

    # Reduced outcome functions
    # Set measurement indices to an array so we can use for array indexing later
    if num_cond:
        measurement_indices = np.array(
            [i for i in range(num_meas) if i not in conditional_indices], dtype=int
        )
        f_meas_outcome = _partial_outcome_function(tuple(measurement_indices))
        f_cond_outcome = _partial_outcome_function(conditional_indices)
    else:
        measurement_indices = None
        f_meas_outcome = lambda x: x
        f_cond_outcome = lambda x: 0

    # Allocate empty stacked basis matrix and prob vector
    reduced_size = size // cdim
    basis_mat = np.zeros((reduced_size, mdim * mdim * pdim * pdim), dtype=complex)
    probs = np.zeros((cdim, reduced_size), dtype=float)

    # Fill matrices
    cond_idxs = {i: 0 for i in range(cdim)}
    for i in range(bsize):
        midx = measurement_data[i]
        midx_meas = midx[measurement_indices] if num_cond else midx
        pidx = preparation_data[i]
        shots = shot_data[i]
        odata = outcome_data[i]

        # Get prep basis component
        if preparation_basis:
            p_mat = np.transpose(preparation_basis.matrix(pidx, preparation_qubits))
        else:
            p_mat = None

        # Get probabilities and optional measurement basis component
        midx_meas = midx[measurement_indices] if num_cond else midx
        meas_cache = set()
        for outcome in range(odata.size):
            # Get conditional and measurement outcome values
            outcome_cond = f_cond_outcome(outcome)
            outcome_meas = f_meas_outcome(outcome)
            idx = cond_idxs[outcome_cond]

            # Store probability
            probs[outcome_cond, idx] = odata[outcome] / shots

            # Check if new meas basis element and construct basis matrix
            store_mat = True
            if measurement_basis:
                if outcome_meas not in meas_cache:
                    meas_cache.add(outcome_meas)
                    mat = measurement_basis.matrix(midx_meas, outcome_meas, measurement_qubits)
                    if preparation_basis:
                        mat = np.kron(p_mat, mat)
                else:
                    store_mat = False
            else:
                mat = p_mat
            if store_mat:
                basis_mat[idx] = np.conj(np.ravel(mat, order="F"))

            # Increase counter
            cond_idxs[outcome_cond] += 1

    return basis_mat, probs


def binomial_weights(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    beta: float = 0,
    conditional_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    r"""Compute weights vector from the binomial distribution.

    The returned weights are given by :math:`w_i = 1 / \sigma_i` where
    the standard deviation :math:`\sigma_i` is estimated as
    :math:`\sigma_i = \sqrt{p_i(1-p_i) / n_i}`. To avoid dividing
    by zero the probabilities are hedged using the *add-beta* rule

    .. math:
        p_i = \frac{f_i + \beta}{n_i + K \beta}

    where :math:`f_i` is the observed frequency, :math:`n_i` is the
    number of shots, and :math:`K` is the number of possible measurement
    outcomes.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        beta: Hedging parameter for converting frequencies to
              probabilities. If 0 hedging is disabled.
        conditional_indices: outcome indices of conditional outcome data.

    Returns:
        The weight vector.
    """
    size = outcome_data.size
    num_data, num_outcomes = outcome_data.shape
    if conditional_indices:
        dim_cond = 2 ** len(conditional_indices)
        f_cond_outcome = _partial_outcome_function(tuple(conditional_indices))
    else:
        dim_cond = 1
        f_cond_outcome = lambda x: 0

    # Compute hedged probabilities where the "add-beta" rule ensures
    # there are no zero or 1 values so we don't have any zero variance
    probs = np.zeros((dim_cond, size // dim_cond), dtype=float)
    prob_shots = np.zeros((dim_cond, size // dim_cond), dtype=int)

    # Fill matrices
    cond_idxs = {i: 0 for i in range(dim_cond)}
    for i in range(num_data):
        shots = shot_data[i]
        denom = shots + num_outcomes * beta
        freqs = outcome_data[i]
        for outcome in range(num_outcomes):
            outcome_cond = f_cond_outcome(outcome)
            idx = cond_idxs[outcome_cond]
            probs[outcome_cond, idx] = (freqs[outcome] + beta) / denom
            prob_shots[outcome_cond, idx] = shots
            cond_idxs[outcome_cond] += 1
    variance = probs * (1 - probs)
    return np.sqrt(prob_shots / variance)


@functools.lru_cache(None)
def _partial_outcome_function(indices: Tuple[int]) -> Callable:
    """Return function for computing partial outcome of specified indices"""
    # NOTE: This function only works for 2-outcome subsystem measurements
    ind_array = np.asarray(indices, dtype=int)
    mask_array = 1 << ind_array
    bit_array = 1 << np.arange(ind_array.size, dtype=int)

    @functools.lru_cache(None)
    def partial_outcome(outcome: int) -> int:
        return np.dot(bit_array, (mask_array & outcome) >> ind_array)

    return partial_outcome
