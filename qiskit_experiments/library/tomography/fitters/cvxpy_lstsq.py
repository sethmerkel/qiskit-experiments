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
Contrained convex least-squares tomography fitter.
"""

from typing import Optional, Dict, Sequence, Tuple
import numpy as np

from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
)
from . import cvxpy_utils
from .cvxpy_utils import cvxpy
from . import lstsq_utils


@cvxpy_utils.requires_cvxpy
def cvxpy_linear_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int]] = None,
    preparation_qubits: Optional[Tuple[int]] = None,
    conditional_indices: Optional[Tuple[int]] = None,
    conditional_matrix: Optional[np.ndarray] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    r"""Constrained weighted linear least-squares tomography fitter.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
                &= \mbox{argmin }\sum_i w_i^2(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2

        subject to

        - *Positive-semidefinite* (``psd=True``): :math:`\rho \gg 0` is constrained
          to be a postive-semidefinite matrix.
        - *Trace* (``trace=t``): :math:`\mbox{Tr}(\rho) = t` is constained to have
          the specified trace.
        - *Trace preserving* (``trace_preserving=True``): When performing process
          tomography the Choi-state :math:`\rho` represents is contstained to be
          trace preserving.

        where

        - :math:`A` is the matrix of measurement operators
          :math:`A = \sum_i |i\rangle\!\langle\!\langle M_i|`
        - :math:`y` is the vector of expectation value data for each projector
          corresponding to estimates of :math:`b_i = Tr[M_i \cdot x]`.
        - :math:`x` is the vectorized density matrix (or Choi-matrix) to be fitted
          :math:`x = |\rho\rangle\\!\rangle`.

    .. note:

        Various solvers can be called in CVXPY using the `solver` keyword
        argument. When ``psd=True`` the optimization problem is a case of a
        *semidefinite program* (SDP) and requires a SDP compatible solver
        for CVXPY. CVXPY includes an SDP compatible solver `SCS`` but it
        is recommended to install the the open-source ``CVXOPT`` solver
        or one of the supported commercial solvers. See the `CVXPY
        documentation
        <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
        for more information on solvers.

    .. note::

        Linear least-squares constructs the full basis matrix :math:`A` as a dense
        numpy array so should not be used for than 5 or 6 qubits. For larger number
        of qubits try the
        :func:`~qiskit_experiments.library.tomography.fitters.linear_inversion`
        fitter function.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
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
        conditional_matrix: Optional matrix of coefficients for combining
                            conditional state components in fit.
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be trace preserving when
                          fitting a Choi-matrix in quantum process
                          tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        weights: Optional array of weights for least squares objective.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPy is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    basis_matrix, probability_data = lstsq_utils.lstsq_data(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        conditional_indices=conditional_indices,
    )

    # Since CVXPY only works with real variables we must specify the real
    # and imaginary parts of matrices seperately: rho = rho_r + 1j * rho_i

    num_components = probability_data.shape[0]
    dim = int(np.sqrt(basis_matrix.shape[1]))

    # Generate list of conditional components for block diagonal matrix
    # rho = sum_k |k><k| \otimes rho(k)
    rhos_r = []
    rhos_i = []
    cons = []
    for _ in range(num_components):
        rho_r, rho_i, _cons = cvxpy_utils.complex_matrix_variable(dim, hermitian=True, psd=psd)
        rhos_r.append(rho_r)
        rhos_i.append(rho_i)
        cons += _cons

    # Add trace constraint. This contraint applies to the sum of the conditional
    # components
    if trace:
        cons += cvxpy_utils.trace_constaint(rhos_r, rhos_i, trace=trace, hermitian=True)

    # Trace preserving constraint when fitting Choi-matrices for
    # quantum process tomography. Note that this adds an implicity
    # trace constraint of trace(rho) = sqrt(len(rho)) = dim
    # if a different trace constraint is specified above this will
    # cause the fitter to fail.
    if trace_preserving:
        if not preparation_qubits:
            preparation_qubits = tuple(range(preparation_data.shape[1]))
        input_dim = np.prod(preparation_basis.matrix_shape(preparation_qubits))
        cons += cvxpy_utils.trace_preserving_constaint(rhos_r, rhos_i, input_dim=input_dim, hermitian=True)

    # OBJECTIVE FUNCTION

    # The function we wish to minimize is || arg ||_2 where
    #   arg =  bm * vec(rho) - data
    # Since we are working with real matrices in CVXPY we expand this as
    #   bm * vec(rho) = (bm_r + 1j * bm_i) * vec(rho_r + 1j * rho_i)
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    #                   + 1j * (bm_r * vec(rho_i) + bm_i * vec(rho_r))
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    # where we drop the imaginary part since the expectation value is real

    # Construct block diagonal fit variable from conditional components
    # Construct objective function
    if weights is not None:
        weights = weights / np.sqrt(np.sum(weights**2))
        probability_data = weights * probability_data

    if num_components > 1:
        if weights is None:
            bm_r = np.real(basis_matrix)
            bm_i = np.imag(basis_matrix)
            bms_r = [bm_r] * num_components
            bms_i = [bm_i] * num_components
        else:
            bms_r = []
            bms_i = []
            for k in range(num_components):
                weighted_mat = weights[k][:, None] * basis_matrix
                bms_r.append(np.real(weighted_mat))
                bms_i.append(np.imag(weighted_mat))

        # Stack lstsq objective from sum of components
        if conditional_matrix is None:
            vecs_r = [cvxpy.vec(rhos_r[k]) for k in range(num_components)]
            vecs_i = [cvxpy.vec(rhos_i[k]) for k in range(num_components)]
        else:
            # Use conditional weights to combine states
            vecs_r = [
                cvxpy.vec(
                    cvxpy.sum([conditional_matrix[j, k] * rhos_r[k] for k in range(num_components)])
                )
                for j in range(num_components)
            ]
            vecs_i = [
                cvxpy.vec(
                    cvxpy.sum([conditional_matrix[j, k] * rhos_i[k] for k in range(num_components)])
                )
                for j in range(num_components)
            ]
        args = [
            bms_r[k] @ vecs_r[k] - bms_i[k] @ vecs_i[k] - probability_data[k]
            for k in range(num_components)
        ]
        arg = cvxpy.hstack(args)
    else:
        # Non-conditional fitting
        if weights is not None:
            basis_matrix = weights[0][:, None] * basis_matrix
        bm_r = np.real(basis_matrix)
        bm_i = np.imag(basis_matrix)
        arg = bm_r @ cvxpy.vec(rhos_r[0]) - bm_i @ cvxpy.vec(rhos_i[0]) - probability_data[0]

    # Optimization problem
    obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))
    prob = cvxpy.Problem(obj, cons)

    # Solve SDP
    cvxpy_utils.set_default_sdp_solver(kwargs)
    cvxpy_utils.solve_iteratively(prob, 5000, **kwargs)

    # Return optimal values and problem metadata
    metadata = {
        "cvxpy_solver": prob.solver_stats.solver_name,
        "cvxpy_status": prob.status,
    }
    if trace_preserving:
        metadata["tp_constraint"] = True
    if psd:
        metadata["psd_constraint"] = True
    if trace:
        metadata["trace_constraint"] = trace

    fits = [rhos_r[k].value + 1j * rhos_i[k].value for k in range(num_components)]
    return fits, metadata


@cvxpy_utils.requires_cvxpy
def cvxpy_gaussian_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int]] = None,
    preparation_qubits: Optional[Tuple[int]] = None,
    conditional_indices: Optional[Tuple[int]] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    **kwargs,
) -> Dict:
    r"""Constrained Gaussian linear least-squares tomography fitter.

    .. note::

        This function calls :func:`cvxpy_linear_lstsq` with a Gaussian weights
        vector. Refer to its documentation for additional details.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= \mbox{argmin} (-\log\mathcal{L}{\rho}) \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2 \\
            -\log\mathcal{L}(\rho)
                &= |W(Ax -y) \|_2^2 \\
                &= \sum_i \frac{1}{\sigma_i^2}(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2

    Additional Details
        The Gaussian weights are estimated from the observed frequency and shot data
        using

        .. math::

            \sigma_i &= \sqrt{\frac{q_i(1 - q_i)}{n_i}} \\
            q_i &= \frac{f_i + \beta}{n_i + K \beta}

        where :math:`q_i` are hedged probabilities which are rescaled to avoid
        0 and 1 values using the "add-beta" rule, with :math:`\beta=0.5`, and
        :math:`K=2^m` the number of measurement outcomes for each basis measurement.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
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
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPY is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    weights = lstsq_utils.binomial_weights(
        outcome_data, shot_data, beta=0.5, conditional_indices=conditional_indices
    )
    return cvxpy_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        conditional_indices=conditional_indices,
        psd=psd,
        trace=trace,
        trace_preserving=trace_preserving,
        weights=weights,
        **kwargs,
    )


@cvxpy_utils.requires_cvxpy
def cvxpy_conditional_linear_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int]] = None,
    preparation_qubits: Optional[Tuple[int]] = None,
    conditional_indices: Optional[Tuple[int]] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict:
    r"""Constrained Gaussian linear least-squares tomography fitter.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
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
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        weights: Optional array of weights for least squares objective.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPY is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    conditional_matrix = _conditional_basis_matrix(
        measurement_basis=measurement_basis,
        measurement_qubits=measurement_qubits,
        conditional_indices=conditional_indices,
    )
    return cvxpy_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        conditional_indices=conditional_indices,
        conditional_matrix=conditional_matrix,
        psd=psd,
        trace=trace,
        trace_preserving=trace_preserving,
        weights=weights,
        **kwargs,
    )


@cvxpy_utils.requires_cvxpy
def cvxpy_conditional_gaussian_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int]] = None,
    preparation_qubits: Optional[Tuple[int]] = None,
    conditional_indices: Optional[Tuple[int]] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    **kwargs,
) -> Dict:
    r"""Constrained conditional Gaussian linear least-squares tomography fitter.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
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
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPY is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    weights = lstsq_utils.binomial_weights(
        outcome_data, shot_data, beta=0.5, conditional_indices=conditional_indices
    )
    return cvxpy_conditional_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        conditional_indices=conditional_indices,
        psd=psd,
        trace=trace,
        trace_preserving=trace_preserving,
        weights=weights,
        **kwargs,
    )


def _conditional_basis_matrix(
    measurement_basis: MeasurementBasis,
    measurement_qubits: Sequence[int],
    conditional_indices: Sequence[int],
) -> np.ndarray:
    """Return matrix of conditional basis element diagonals."""
    if not conditional_indices:
        return np.eye(1, dtype=float)

    conditional_qubits = tuple(measurement_qubits[i] for i in conditional_indices)
    num_cond = len(conditional_indices)
    cond_size = measurement_basis.matrix_shape(conditional_qubits)[0]
    basis_mat_cond = np.eye(cond_size, dtype=float)
    for outcome in range(2**num_cond):
        basis_mat_cond[outcome] = np.real(
            np.diag(measurement_basis.matrix((0,) * num_cond, outcome, conditional_qubits))
        )
    return basis_mat_cond
