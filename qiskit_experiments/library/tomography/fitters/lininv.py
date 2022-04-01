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
Linear inversion MLEtomography fitter.
"""

from typing import Dict, Tuple, Optional, Sequence, List
from functools import lru_cache
import numpy as np
from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
    LocalMeasurementBasis,
    LocalPreparationBasis,
)
from .lstsq_utils import _partial_outcome_function


def linear_inversion(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int]] = None,
    preparation_qubits: Optional[Tuple[int]] = None,
    conditional_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    r"""Linear inversion tomography fitter.

    Overview
        This fitter uses linear inversion to reconstructs the maximum-likelihood
        estimate of the least-squares log-likelihood function

        .. math::
            \hat{\rho}
                &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
                &= \mbox{argmin }\sum_i (\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
                &= \mbox{argmin }\|Ax - y \|_2^2

        where

        * :math:`A = \sum_j |j \rangle\!\langle\!\langle E_j|` is the matrix of measured
          basis elements.
        * :math:`y = \sum_j \hat{p}_j |j\rangle` is the vector of estimated measurement
          outcome probabilites for each basis element.
        * :math:`x = |\rho\rangle\!\rangle` is the vectorized density matrix.

    Additional Details
        The linear inversion solution is given by

        .. math::
            \hat{\rho} = \sum_i \hat{p}_i D_i

        where measurement probabilities :math:`\hat{p}_i = f_i / n_i` are estimated
        from the observed count frequencies :math:`f_i` in :math:`n_i` shots for each
        basis element :math:`i`, and :math:`D_i` is the *dual basis* element constructed
        from basis :math:`\{E_i\}` via:

        .. math:

            |D_i\rangle\!\rangle = M^{-1}|E_i \rangle\!\rangle \\
            M = \sum_j |E_j\rangle\!\rangle\!\langle\!\langle E_j|

    .. note::

        The Linear inversion fitter treats the input measurement and preparation
        bases as local bases and constructs separate 1-qubit dual basis for each
        individual qubit.

        Linear inversion is only possible if the input bases are local and a spanning
        set for the vector space of the reconstructed matrix
        (*tomographically complete*). If the basis is not tomographically complete
        the :func:`~qiskit_experiments.library.tomography.fitters.scipy_linear_lstsq`
        or :func:`~qiskit_experiments.library.tomography.fitters.cvxpy_linear_lstsq`
        function can be used to solve the same objective function via
        least-squares optimization.

    Args:
        outcome_data: basis outcome frequency data.
        shot_data: basis outcome total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: the tomography measurement basis.
        preparation_basis: the tomography preparation basis.
        measurement_qubits: Optional, the physical qubits that were measured.
                            If None they are assumed to be [0, ..., M-1] for
                            M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
                            If None they are assumed to be [0, ..., N-1] for
                            N preparated qubits.
        conditional_indices: Optional, conditional measurement data indices.
                             If set this will return a list of conditional
                             fitted states conditioned on a fixed basis
                             measurement of these qubits.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho.
    """
    # Construct dual bases
    meas_dual_basis = None
    if measurement_basis:
        if not measurement_qubits:
            measurement_qubits = tuple(range(measurement_data.shape[1]))
        meas_duals = {i: _dual_povms(measurement_basis, i) for i in measurement_qubits}
        meas_dual_basis = LocalMeasurementBasis(
            f"Dual_{measurement_basis.name}", qubit_povms=meas_duals
        )

    prep_dual_basis = None
    if preparation_basis:
        if not preparation_qubits:
            preparation_qubits = tuple(range(preparation_data.shape[1]))
        prep_duals = {i: _dual_states(preparation_basis, i) for i in preparation_qubits}
        prep_dual_basis = LocalPreparationBasis(
            f"Dual_{preparation_basis.name}", qubit_states=prep_duals
        )

    if shot_data is None:
        shot_data = np.ones(len(outcome_data))

    if conditional_indices:
        # Split measurement qubits into conditional and non-conditional qubits
        f_cond_qubits = []
        f_meas_qubits = []
        f_meas_indices = []
        for i, qubit in enumerate(measurement_qubits):
            if i in conditional_indices:
                f_cond_qubits.append(qubit)
            else:
                f_meas_qubits.append(qubit)
                f_meas_indices.append(i)

        # Get size of conditional outcomes
        cond_size = np.prod(measurement_basis.outcome_shape(f_cond_qubits))

        # Indexing array for fully tomo measured qubits
        f_meas_indices = np.array(f_meas_indices, dtype=int)

        # Reduced outcome functions
        f_meas_outcome = _partial_outcome_function(tuple(f_meas_indices))
        f_cond_outcome = _partial_outcome_function(tuple(conditional_indices))
    else:
        cond_size = 1
        f_meas_qubits = measurement_qubits
        f_meas_indices = slice(None)
        f_meas_outcome = lambda x: x
        f_cond_outcome = lambda x: 0

    # Calculate shape of matrix to be fitted
    if preparation_qubits:
        pdim = np.prod(prep_dual_basis.matrix_shape(preparation_qubits))
    else:
        pdim = 1
    if measurement_basis:
        mdim = np.prod(meas_dual_basis.matrix_shape(f_meas_qubits))
    else:
        mdim = 1
    shape = (pdim * mdim, pdim * mdim)

    # Construct linear inversion matrix
    # This does not currently support readout error mitigation of
    # conditional bit measurements
    fits = [np.zeros(shape, dtype=complex) for _ in range(cond_size)]
    for i, outcomes in enumerate(outcome_data):
        shots = shot_data[i]
        pidx = preparation_data[i]
        midx = measurement_data[i][f_meas_indices]

        # Get prep basis component
        if prep_dual_basis:
            p_mat = np.transpose(prep_dual_basis.matrix(pidx, preparation_qubits))
        else:
            p_mat = None

        # Get probabilities and optional measurement basis component
        for outcome, freq in enumerate(outcomes):
            if freq == 0:
                # Skip component with zero probability
                continue
            prob = freq / shots

            # Get component on non-conditional bits
            outcome_meas = f_meas_outcome(outcome)
            if meas_dual_basis:
                dual_op = meas_dual_basis.matrix(midx, outcome_meas, f_meas_qubits)
                if prep_dual_basis:
                    dual_op = np.kron(p_mat, dual_op)
            else:
                dual_op = p_mat

            # Add component to correct conditional
            outcome_cond = f_cond_outcome(outcome)
            fits[outcome_cond] += prob * dual_op

    return fits, {}


@lru_cache(None)
def _dual_states(basis: PreparationBasis, qubit: int) -> np.ndarray:
    """Construct a dual preparation basis for linear inversion"""
    size = basis.index_shape((qubit,))[0]
    states = np.asarray([basis.matrix((i,), (qubit,)) for i in range(size)])
    return _construct_dual_states(states)


@lru_cache(None)
def _dual_povms(basis: MeasurementBasis, qubit: int) -> List[List[np.ndarray]]:
    """Construct dual POVM states for linear inversion"""
    size = basis.index_shape((qubit,))[0]
    num_outcomes = basis.outcome_shape((qubit,))[0]

    # Concatenate all POVM effects to treat as states for linear inversion
    states = []
    for index in range(size):
        for outcome in range(num_outcomes):
            states.append(basis.matrix((index,), outcome, (qubit,)))

    dual_basis = _construct_dual_states(states)

    # Organize back into nested lists of dual POVM effects
    dual_povms = []
    idx = 0
    for _ in range(size):
        dual_povms.append([dual_basis[idx + i] for i in range(num_outcomes)])
        idx += num_outcomes
    return dual_povms


def _construct_dual_states(states: Sequence[np.ndarray]):
    """Construct a dual preparation basis for linear inversion"""
    mats = np.asarray(states)
    size, dim1, dim2 = np.shape(mats)
    vec_basis = np.reshape(mats, (size, dim1 * dim2))
    basis_mat = np.sum([np.outer(i, np.conj(i)) for i in vec_basis], axis=0)

    try:
        inv_mat = np.linalg.inv(basis_mat)
    except np.linalg.LinAlgError as ex:
        raise ValueError(
            "Cannot construct dual basis states. Input states are not tomographically complete"
        ) from ex

    vec_dual = np.tensordot(inv_mat, vec_basis, axes=([1], [1])).T
    dual_mats = np.reshape(vec_dual, (size, dim1, dim2)).round(15)
    return dual_mats
