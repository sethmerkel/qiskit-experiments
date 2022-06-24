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
Quantum process tomography analysis
"""


from typing import List, Dict, Tuple, Union, Optional, Callable
import functools
import time
import numpy as np
import scipy.linalg as la

from qiskit.result import marginal_counts, Counts
from qiskit.quantum_info import DensityMatrix, Choi, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options
from .basis import MeasurementBasis
from .fitters import (
    linear_inversion,
    scipy_linear_lstsq,
    scipy_gaussian_lstsq,
    cvxpy_linear_lstsq,
    cvxpy_gaussian_lstsq,
    cvxpy_conditional_linear_lstsq,
    cvxpy_conditional_gaussian_lstsq,
)


class TomographyAnalysis(BaseAnalysis):
    """Base analysis for state and process tomography experiments."""

    _builtin_fitters = {
        "linear_inversion": linear_inversion,
        "scipy_linear_lstsq": scipy_linear_lstsq,
        "scipy_gaussian_lstsq": scipy_gaussian_lstsq,
        "cvxpy_linear_lstsq": cvxpy_linear_lstsq,
        "cvxpy_gaussian_lstsq": cvxpy_gaussian_lstsq,
        "cvxpy_conditional_linear_lstsq": cvxpy_conditional_linear_lstsq,
        "cvxpy_conditional_gaussian_lstsq": cvxpy_conditional_gaussian_lstsq,
    }
    _cvxpy_fitters = (
        cvxpy_linear_lstsq,
        cvxpy_gaussian_lstsq,
        cvxpy_conditional_linear_lstsq,
        cvxpy_conditional_gaussian_lstsq,
    )

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options

        Analysis Options:
            measurement_basis
                (:class:`~qiskit_experiments.library.tomography.basis.MeasurementBasis`):
                The measurement
                :class:`~qiskit_experiments.library.tomography.basis.MeasurementBasis`
                to use for tomographic reconstruction when running a
                :class:`~qiskit_experiments.library.tomography.StateTomography` or
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            preparation_basis
                (:class:`~qiskit_experiments.library.tomography.basis.PreparationBasis`):
                The preparation
                :class:`~qiskit_experiments.library.tomography.basis.PreparationBasis`
                to use for tomographic reconstruction for
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            fitter (str or Callable): The fitter function to use for reconstruction.
                This can  be a string to select one of the built-in fitters, or a callable to
                supply a custom fitter function. See the `Fitter Functions` section for
                additional information.
            fitter_options (dict): Any addition kwarg options to be supplied to the fitter
                function. For documentation of available kargs refer to the fitter function
                documentation.
            rescale_positive (bool): If True rescale the state returned by the fitter
                to be positive-semidefinite. See the `PSD Rescaling` section for
                additional information (Default: True).
            rescale_trace (bool): If True rescale the state returned by the fitter
                have either trace 1 for :class:`~qiskit.quantum_info.DensityMatrix`,
                or trace dim for :class:`~qiskit.quantum_info.Choi` matrices (Default: True).
            target (Any): Optional, target object for fidelity comparison of the fit
                (Default: None).
            conditional_indices (list[int]): Optional, indices of conditional measurement
                qubits for reconstructing a list of conditional states on the remaining
                measurement qubits.
        """
        options = super()._default_options()

        options.measurement_basis = None
        options.preparation_basis = None
        options.measurement_qubits = None
        options.preparation_qubits = None
        options.fitter = "linear_inversion"
        options.fitter_options = {}
        options.rescale_positive = True
        options.rescale_trace = True
        options.target = None
        options.conditional_indices = None
        return options

    @classmethod
    def _get_fitter(cls, fitter: Union[str, Callable]) -> Callable:
        """Return fitter function for named builtin fitters"""
        if fitter is None:
            raise AnalysisError("No tomography fitter given")
        if not isinstance(fitter, str):
            return fitter
        if fitter in cls._builtin_fitters:
            return cls._builtin_fitters[fitter]
        raise AnalysisError(f"Unrecognized tomography fitter {fitter}")

    def _run_analysis(self, experiment_data):
        # Get option values.
        measurement_basis = self.options.measurement_basis
        measurement_qubits = self.options.measurement_qubits
        if measurement_basis and measurement_qubits is None:
            measurement_qubits = experiment_data.metadata.get("m_qubits")
        preparation_basis = self.options.preparation_basis
        preparation_qubits = self.options.preparation_qubits
        if preparation_basis and preparation_qubits is None:
            preparation_qubits = experiment_data.metadata.get("p_qubits")
        conditional_indices = self.options.conditional_indices
        if conditional_indices is None:
            conditional_indices = experiment_data.metadata.get("c_indices")

        # Generate tomography fitter data
        outcome_data, shot_data, measurement_data, preparation_data = self._fitter_data(
            experiment_data.data(),
            measurement_basis=measurement_basis,
            measurement_qubits=measurement_qubits,
        )

        # Construct default values for qubit options if not provided
        if preparation_qubits is None:
            preparation_qubits = tuple(range(preparation_data.shape[1]))
        if measurement_qubits is None:
            measurement_qubits = tuple(range(measurement_data.shape[1]))

        # Get dimension of the preparation and measurement qubits subystems
        prep_dims = (1,)
        if preparation_qubits:
            prep_dims = preparation_basis.matrix_shape(preparation_qubits)
        meas_dims = (1,)
        full_meas_qubits = measurement_qubits
        if measurement_qubits:
            if conditional_indices is not None:
                # Remove conditional qubits from full meas qubits
                full_meas_qubits = [
                    q for i, q in enumerate(measurement_qubits) if i not in conditional_indices
                ]
            if full_meas_qubits:
                meas_dims = measurement_basis.matrix_shape(full_meas_qubits)

        if full_meas_qubits:
            # QPT or QST
            input_dims = prep_dims
            output_dims = meas_dims
        else:
            # QST of POVM effects
            input_dims = meas_dims
            output_dims = prep_dims

        # Use preparation dim to set the expected trace of the fitted state.
        # For QPT this is the input dimension, for QST this will always be 1.
        trace = np.prod(prep_dims) if self.options.rescale_trace else None

        # Get tomography fitter function
        fitter = self._get_fitter(self.options.fitter)
        fitter_opts = self.options.fitter_options

        # Work around to set proper trace and trace preserving constraints for
        # cvxpy fitter
        if fitter in self._cvxpy_fitters:
            fitter_opts = fitter_opts.copy()

            # Add default value for CVXPY trace constraint if no user value is provided
            # Use preparation dim to set the expected trace of the fitted state.
            # For QPT this is the input dimension, for QST this will always be 1.
            if "trace" not in fitter_opts:
                fitter_opts["trace"] = trace

            # By default add trace preserving constraint to cvxpy QPT fit
            if preparation_data.shape[1] > 0 and "trace_preserving" not in fitter_opts:
                fitter_opts["trace_preserving"] = True

        # Run tomography fitter
        t_fitter_start = time.time()
        try:
            fit, fitter_metadata = fitter(
                outcome_data,
                shot_data,
                measurement_data,
                preparation_data,
                measurement_basis=measurement_basis,
                preparation_basis=preparation_basis,
                measurement_qubits=measurement_qubits,
                preparation_qubits=preparation_qubits,
                conditional_indices=conditional_indices,
                **fitter_opts,
            )
        except AnalysisError as ex:
            raise AnalysisError(f"Tomography fitter failed with error: {str(ex)}") from ex
        t_fitter_stop = time.time()

        # Add fitter metadata
        if fitter_metadata is None:
            fitter_metadata = {}
        fitter_metadata["fitter"] = fitter.__name__
        fitter_metadata["fitter_time"] = t_fitter_stop - t_fitter_start

        # Post process fit
        analysis_results = self._postprocess_fit(
            fit,
            fitter_metadata=fitter_metadata,
            trace=trace,
            make_positive=self.options.rescale_positive,
            input_dims=input_dims,
            output_dims=output_dims,
            target_state=self.options.target,
        )
        return analysis_results, []

    @classmethod
    def _postprocess_fit(
        cls,
        fits: List[np.ndarray],
        fitter_metadata: Optional[Dict] = None,
        trace: Optional[float] = None,
        make_positive: bool = False,
        input_dims: Optional[Tuple[int]] = None,
        output_dims: Optional[Tuple[int]] = None,
        target_state: Optional[Union[Choi, DensityMatrix]] = None,
    ) -> Dict[str, any]:
        """Post-process raw fitter result"""
        # Convert fitter matrix to state data for post-processing
        input_dim = np.prod(input_dims) if input_dims else 1
        qpt = input_dim > 1
        state_results = [
            cls._state_result(
                fit,
                make_positive=make_positive,
                trace=trace,
                input_dims=input_dims,
                output_dims=output_dims,
                fitter_metadata=fitter_metadata,
            )
            for fit in fits
        ]

        # Compute the conditional probability of each component so that the
        # total probability of all components is 1, and optional rescale trace
        # of each component
        fit_traces = [res.extra.pop("fit_trace") for res in state_results]
        total_trace = sum(fit_traces)
        for i, (fit_trace, res) in enumerate(zip(fit_traces, state_results)):
            # Compute conditional component probability from the the component
            # non-rescaled fit trace
            res.extra["component_probability"] = fit_trace / total_trace
            res.extra["component_index"] = i

        other_results = []
        # Compute fidelity with target
        if len(state_results) == 1 and target_state is not None:
            # Note: this currently only works for non-conditional tomography
            other_results.append(
                cls._fidelity_result(state_results[0], target_state, input_dim=input_dim)
            )

        # Check positive
        other_results.append(cls._positivity_result(state_results, qpt=qpt))

        # Check trace preserving
        if qpt:
            other_results.append(cls._tp_result(state_results, input_dim=input_dim))

        # Finally format state result metadata to remove eigenvectors
        # which are no longer needed to reduce size
        for state_result in state_results:
            state_result.extra.pop("eigvecs")

        return state_results + other_results

    @classmethod
    def _state_result(
        cls,
        fit: np.ndarray,
        make_positive: bool = False,
        trace: Optional[float] = None,
        input_dims: Optional[Tuple[int]] = None,
        output_dims: Optional[Tuple[int]] = None,
        fitter_metadata: Optional[Dict] = None,
    ) -> List[AnalysisResultData]:
        """Convert fit data to state result data"""
        # Get eigensystem of state fit
        raw_eigvals, eigvecs = cls._state_eigensystem(fit)

        # Optionally rescale eigenvalues to be non-negative
        if make_positive and np.any(raw_eigvals < 0):
            eigvals = cls._make_positive(raw_eigvals)
            fit = eigvecs @ (eigvals * eigvecs).T.conj()
            rescaled_psd = True
        else:
            eigvals = raw_eigvals
            rescaled_psd = False

        # Optionally rescale fit trace
        fit_trace = np.sum(eigvals)
        if trace is not None and not np.isclose(fit_trace - trace, 0, atol=1e-12):
            scale = trace / fit_trace
            fit = fit * scale
            eigvals = eigvals * scale
        else:
            trace = fit_trace

        # Convert class of value
        if input_dims and np.prod(input_dims) > 1:
            value = Choi(fit, input_dims=input_dims, output_dims=output_dims)
        else:
            value = DensityMatrix(fit, dims=output_dims)

        # Construct state result extra metadata
        extra = {
            "trace": trace,
            "eigvals": eigvals,
            "raw_eigvals": raw_eigvals,
            "rescaled_psd": rescaled_psd,
            "fit_trace": fit_trace,
            "eigvecs": eigvecs,
            "fitter_metadata": fitter_metadata or {},
        }
        return AnalysisResultData("state", value, extra=extra)

    @staticmethod
    def _positivity_result(
        state_results: List[AnalysisResultData], qpt: bool = False
    ) -> AnalysisResultData:
        """Check if eigenvalues are positive"""
        total_cond = 0.0
        comps_cond = []
        comps_pos = []
        name = "completely_positive" if qpt else "positive"
        for result in state_results:
            evals = result.extra["eigvals"]

            # Check if component is positive and add to extra if so
            cond = np.sum(np.abs(evals[evals < 0]))
            pos = bool(np.isclose(cond, 0))
            result.extra[name] = pos

            # Add component to combined result
            comps_cond.append(cond)
            comps_pos.append(pos)
            total_cond += cond * result.extra["component_probability"]

        # Check if combined conditional state is positive
        is_pos = bool(np.isclose(total_cond, 0))
        result = AnalysisResultData(name, is_pos)
        if not is_pos:
            result.extra = {
                "delta": total_cond,
                "components": comps_pos,
                "components_delta": comps_cond,
            }
        return result

    @staticmethod
    def _tp_result(
        state_results: List[AnalysisResultData],
        input_dim: int = 1,
    ) -> AnalysisResultData:
        """Check if QPT channel is trace preserving"""
        # Construct the Kraus TP condition matrix sum_i K_i^dag K_i
        # summed over all components k
        kraus_cond = 0.0
        for result in state_results:
            evals = result.extra["eigvals"]
            evecs = result.extra["eigvecs"]
            prob = result.extra["component_probability"]
            size = len(evals)
            output_dim = size // input_dim
            mats = np.reshape(evecs.T, (size, output_dim, input_dim), order="F")
            comp_cond = np.einsum("i,ija,ijb->ab", evals, mats.conj(), mats)
            kraus_cond = kraus_cond + prob * comp_cond

        tp_cond = np.sum(np.abs(la.eigvalsh(kraus_cond - np.eye(input_dim))))
        is_tp = bool(np.isclose(tp_cond, 0))
        result = AnalysisResultData("trace_preserving", is_tp)
        if not is_tp:
            result.extra = {"delta": tp_cond}
        return result

    @staticmethod
    def _fidelity_result(
        state_result: AnalysisResultData,
        target: Union[Choi, DensityMatrix],
        input_dim: int = 1,
    ) -> AnalysisResultData:
        """Faster computation of fidelity from eigen decomposition"""
        evals = state_result.extra["eigvals"]
        evecs = state_result.extra["eigvecs"]

        # Format target to statevector or densitymatrix array
        name = "process_fidelity" if input_dim > 1 else "state_fidelity"
        if target is None:
            raise AnalysisError("No target state provided")
        if isinstance(target, QuantumChannel):
            target_state = Choi(target).data / input_dim
        elif isinstance(target, BaseOperator):
            target_state = np.ravel(Operator(target), order="F") / np.sqrt(input_dim)
        else:
            # Statevector or density matrix
            target_state = np.array(target)

        if target_state.ndim == 1:
            rho = evecs @ (evals / input_dim * evecs).T.conj()
            fidelity = np.real(target_state.conj() @ rho @ target_state)
        else:
            sqrt_rho = evecs @ (np.sqrt(evals / input_dim) * evecs).T.conj()
            eig = la.eigvalsh(sqrt_rho @ target_state @ sqrt_rho)
            fidelity = np.sum(np.sqrt(np.maximum(eig, 0))) ** 2
        return AnalysisResultData(name, fidelity)

    @staticmethod
    def _state_eigensystem(fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the eigensystem of the fitted state.

        The eigenvalues are returned as a real array ordered from
        smallest to largest eigenvalues.

        Args:
            fit: the fitted state matrix.

        Returns:
            A pair of (eigenvalues, eigenvectors).
        """
        evals, evecs = la.eigh(fit)
        # Truncate eigenvalues to real part
        evals = np.real(evals)
        # Sort eigensystem from largest to smallest eigenvalues
        sort_inds = np.flip(np.argsort(evals))
        return evals[sort_inds], evecs[:, sort_inds]

    @staticmethod
    def _make_positive(evals: np.ndarray, epsilon: float = 0) -> np.ndarray:
        """Rescale a real vector to be non-negative.

        This truncates any negative values to zero and rescales
        the remaining eigenvectors such that the sum of the vector
        is preserved.
        """
        if epsilon < 0:
            raise AnalysisError("epsilon must be non-negative.")
        ret = evals.copy()
        dim = len(evals)
        idx = dim - 1
        accum = 0.0
        while idx >= 0:
            shift = accum / (idx + 1)
            if evals[idx] + shift < epsilon:
                ret[idx] = 0
                accum = accum + evals[idx]
                idx -= 1
            else:
                for j in range(idx + 1):
                    ret[j] = evals[j] + shift
                break
        return ret

    @staticmethod
    def _fitter_data(
        data: List[Dict[str, any]],
        measurement_basis: Optional[MeasurementBasis] = None,
        measurement_qubits: Optional[Tuple[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return list a tuple of basis, frequency, shot data"""
        meas_size = None
        prep_size = None

        # Construct marginalized tomography count dicts
        outcome_dict = {}
        shots_dict = {}
        for datum in data:
            # Get basis data
            metadata = datum["metadata"]
            meas_element = tuple(metadata["m_idx"]) if "m_idx" in metadata else tuple()
            prep_element = tuple(metadata["p_idx"]) if "p_idx" in metadata else tuple()
            if meas_size is None:
                meas_size = len(meas_element)
            if prep_size is None:
                prep_size = len(prep_element)

            # Add outcomes
            counts = Counts(marginal_counts(datum["counts"], metadata["clbits"]))
            shots = datum.get("shots", sum(counts.values()))
            basis_key = (meas_element, prep_element)
            if basis_key in outcome_dict:
                TomographyAnalysis._append_counts(outcome_dict[basis_key], counts)
                shots_dict[basis_key] += shots
            else:
                outcome_dict[basis_key] = counts
                shots_dict[basis_key] = shots

        # Construct function for converting count outcome dit-strings into
        # integers based on the specified number of outcomes of the measurement
        # bases on each qubit
        if meas_size == 0:
            # Trivial case with no measurement
            num_outcomes = 1
            outcome_func = lambda _: 1
        elif measurement_basis is None:
            # If no basis is provided assume N-qubit measurement case
            num_outcomes = 2**meas_size
            outcome_func = lambda outcome: int(outcome, 2)
        else:
            # General measurement basis case for arbitrary outcome measurements
            if measurement_qubits is None:
                measurement_qubits = tuple(range(meas_size))
            elif len(measurement_qubits) != meas_size:
                raise AnalysisError("Specified number of measurementqubits does not match data.")
            outcome_shape = measurement_basis.outcome_shape(measurement_qubits)
            num_outcomes = np.prod(outcome_shape)
            outcome_func = _int_outcome_function(outcome_shape)

        num_basis = len(outcome_dict)
        measurement_data = np.zeros((num_basis, meas_size), dtype=int)
        preparation_data = np.zeros((num_basis, prep_size), dtype=int)
        shot_data = np.zeros(num_basis, dtype=int)
        outcome_data = np.zeros((num_basis, num_outcomes), dtype=int)

        for i, (basis_key, counts) in enumerate(outcome_dict.items()):
            measurement_data[i] = basis_key[0]
            preparation_data[i] = basis_key[1]
            shot_data[i] = shots_dict[basis_key]
            for outcome, freq in counts.items():
                outcome_data[i][outcome_func(outcome)] = freq
        return outcome_data, shot_data, measurement_data, preparation_data

    @staticmethod
    def _append_counts(counts1, counts2):
        for key, val in counts2.items():
            if key in counts1:
                counts1[key] += val
            else:
                counts1[key] = val
        return counts1


@functools.lru_cache(None)
def _int_outcome_function(outcome_shape: Tuple[int]) -> Callable:
    """Generate function for converting string outcomes to ints"""
    # Recursively extract leading bit(dit)
    if len(set(outcome_shape)) == 1:
        # All outcomes are the same shape, so we can use a constant base
        base = outcome_shape[0]
        return lambda outcome: int(outcome, base)

    # General function where each dit could be a differnet base
    @functools.lru_cache(2048)
    def _int_outcome_general(outcome: str):
        """Convert a general dit-string outcome to integer"""
        # Recursively extract leading bit(dit)
        value = 0
        for i, base in zip(outcome, outcome_shape):
            value *= base
            value += int(i, base)
        return value

    return _int_outcome_general
