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
=============================================================
Database Service (:mod:`qiskit_experiments.database_service`)
=============================================================

.. currentmodule:: qiskit_experiments.database_service

This subpackage provides database-specific utility functions and exceptions which
are used with the :class:`.ExperimentData` and :class:`.AnalysisResult` classes.

Device Components
=================

.. autosummary::
   :toctree: ../stubs/

   DeviceComponent
   Qubit
   Resonator
   UnknownComponent
   to_component

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   ExperimentDataError
   ExperimentEntryExists
   ExperimentEntryNotFound
"""

from .exceptions import ExperimentDataError, ExperimentEntryExists, ExperimentEntryNotFound
from .device_component import DeviceComponent, Qubit, Resonator, UnknownComponent, to_component
