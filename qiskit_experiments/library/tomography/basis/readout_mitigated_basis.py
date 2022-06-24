# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Readout error mitigated tomography basis.
"""
from typing import Sequence, Optional, Tuple
import functools
import numpy as np
from qiskit.result import BaseReadoutMitigator, LocalReadoutMitigator
from qiskit.exceptions import QiskitError
from .base_basis import MeasurementBasis
from .local_basis import LocalMeasurementBasis


class ReadoutMitigatedMeasurementBasis(MeasurementBasis):
    """Readout error mitigated measurement basis.

    This basis returns the noisy POVM elements obtained by applying
    a readout error model from a :class:`.BaseReadoutMitigator`
    to another :class:`.MeasurementBasis`.
    """

    def __init__(
        self,
        basis: MeasurementBasis,
        readout_mitigator: Optional[BaseReadoutMitigator],
    ):
        """Initialize a fitter preparation basis.

        Args:
            basis: the measurement basis to apply readout noise model to.
            readout_mitigator: readout error mitigator for constructing
                               the noisy POVMs from the supplied POVMs based
                               on the mitigators readout error model.

        Raises:
            QiskitError: if input basis or readout mitigator is not valid.
        """
        super().__init__(f"Mitigated_{basis.name}")

        if not isinstance(basis, MeasurementBasis):
            raise QiskitError("Input basis is not a MeasurementBasis")
        if not isinstance(readout_mitigator, BaseReadoutMitigator):
            raise QiskitError("Input mitigator is not a BaseReadoutMitigator")

        self._basis = basis
        self._mitigator = readout_mitigator

        if self._mitigator is None:
            self._povm_fn = self._ideal_povm
        elif isinstance(basis, LocalMeasurementBasis) and isinstance(
            self._mitigator, LocalReadoutMitigator
        ):
            self._povm_fn = self._local_noisy_povm
        else:
            self._povm_fn = self._noisy_povm

    def __hash__(self):
        return hash((type(self), self._name, hash(self._basis), type(self._mitigator)))

    def __eq__(self, value):
        return (
            super().__eq__(value)
            and self._basis == getattr(value, "_basis", None)
            and self._mitigator == getattr(value, "_mitigator", None)
        )

    def __json_encode__(self):
        return {"basis": self._basis, "readout_mitigator": self._mitigator}

    def index_shape(self, qubits: Sequence[int]) -> Tuple[int]:
        return self._basis.index_shape(qubits)

    def outcome_shape(self, qubits: Sequence[int]) -> Tuple[int]:
        return self._basis.outcome_shape(qubits)

    def matrix_shape(self, qubits: Sequence[int]) -> Tuple[int]:
        return self._basis.matrix_shape(qubits)

    def circuit(self, index: Sequence[int], qubits: Sequence[int]):
        return self._basis.circuit(index, qubits)

    def matrix(self, index: Sequence[int], outcome: int, qubits: Sequence[int]) -> np.ndarray:
        return self._povm_fn(index, outcome, qubits)

    def _ideal_povm(self, index: Sequence[int], outcome: int, qubits: Sequence[int]):
        # pylint: disable = unused-argument
        return self._basis.matrix(index, outcome, qubits)

    def _noisy_povm(self, index: Sequence[int], outcome: int, qubits: Sequence[int]):
        """Return the noisy basis POVM element.

        This is calculated by summing the ideal POVM basis elements
        using the readout assignment error probabilities from the
        basis readout error mitigator.
        """
        # Get the readout error values coefficients for combining
        # ideal POVM elements into noisy POVM element
        coeffs = self._mitigator.assignment_matrix(qubits)[outcome]
        ret = 0
        for i, coeff in enumerate(coeffs):
            ret = ret + coeff * self._basis.matrix(index, i, qubits)
        return ret

    def _local_noisy_povm(self, index: Sequence[int], outcome: int, qubits: Sequence[int]):
        """Return the noisy basis POVM element.

        This is calculated by summing the ideal POVM basis elements
        using the readout assignment error probabilities from the
        basis readout error mitigator. It assumes the readout
        errors are local to obtain the probabilities for the
        single-qubit assignment matrices.
        """
        num_index = len(index)
        if num_index == 1:
            return self._noisy_povm_single(index[0], outcome, qubits[0])
        outcome_index = self._basis._outcome_indices(outcome, tuple(qubits))
        ret = 1
        for i in range(num_index):
            local = self._noisy_povm_single(index[i], outcome_index[i], qubits[i])
            ret = np.kron(local, ret)
        return ret

    @functools.lru_cache(None)
    def _noisy_povm_single(self, index: int, outcome: int, qubit: int):
        """Cached version of noisy_povm function for use in _local_noisy_povm"""
        return self._noisy_povm((index,), outcome, (qubit,))
