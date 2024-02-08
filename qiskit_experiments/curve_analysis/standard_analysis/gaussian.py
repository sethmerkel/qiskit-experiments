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

"""Resonance analysis class based on a Gaussian fit."""

from typing import List, Union, Optional

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import Options


class GaussianAnalysis(curve.CurveAnalysis):
    r"""A class to analyze a resonance, typically seen as a peak.

    Overview
        This analysis takes only single series. This series is fit by the Gaussian function.

    Fit Model
        The fit is based on the following Gaussian function.

        .. math::

            F(x) = a \exp(-(x-f)^2/(2\sigma^2)) + b

    Fit Parameters
        - :math:`a`: Peak height.
        - :math:`b`: Base line.
        - :math:`f`: Center frequency. This is the fit parameter of main interest.
        - :math:`\sigma`: Standard deviation of Gaussian function.

    Initial Guesses
        - :math:`a`: Calculated by :func:`~qiskit_experiments.curve_analysis.guess.max_height`.
        - :math:`b`: Calculated by :func:`~qiskit_experiments.curve_analysis.guess.\
          constant_spectral_offset`.
        - :math:`f`: Frequency at max height position calculated by
          :func:`~qiskit_experiments.curve_analysis.guess.max_height`.
        - :math:`\sigma`: Calculated from FWHM of peak :math:`w`
          such that :math:`w / \sqrt{8} \ln{2}`, where FWHM is calculated by
          :func:`~qiskit_experiments.curve_analysis.guess.full_width_half_max`.

    Bounds
        - :math:`a`: [-2, 2] scaled with maximum signal value.
        - :math:`b`: [-1, 1] scaled with maximum signal value.
        - :math:`f`: [min(x), max(x)] of frequency scan range.
        - :math:`\sigma`: [0, :math:`\Delta x`] where :math:`\Delta x`
          represents frequency scan range.

    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a * exp(-(x-freq)**2 / (2*sigma**2)) + b",
                    name="gaussian",
                )
            ],
            name=name,
        )

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plotter.set_figure_options(
            xlabel="Frequency",
            ylabel="Signal (arb. units)",
            xval_unit="Hz",
        )
        options.result_parameters = [curve.ParameterRepr("freq", "f01", "Hz")]
        options.normalization = True
        return options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)

        user_opt.bounds.set_if_empty(
            a=(-2 * max_abs_y, 2 * max_abs_y),
            sigma=(0, np.ptp(curve_data.x)),
            freq=(min(curve_data.x), max(curve_data.x)),
            b=(-max_abs_y, max_abs_y),
        )
        user_opt.p0.set_if_empty(b=curve.guess.constant_spectral_offset(curve_data.y))

        y_ = curve_data.y - user_opt.p0["b"]

        _, peak_idx = curve.guess.max_height(y_, absolute=True)
        fwhm = curve.guess.full_width_half_max(curve_data.x, y_, peak_idx)

        user_opt.p0.set_if_empty(
            a=curve_data.y[peak_idx] - user_opt.p0["b"],
            freq=curve_data.x[peak_idx],
            sigma=fwhm / np.sqrt(8 * np.log(2)),
        )

        return user_opt

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared less than 3 and greater than zero,
            - a peak within the scanned frequency range,
            - a standard deviation that is not larger than the scanned frequency range,
            - a standard deviation that is wider than the smallest frequency increment,
            - a signal-to-noise ratio, defined as the amplitude of the peak divided by the
              square root of the median y-value less the fit offset, greater than a
              threshold of two, and
            - a standard error on the sigma of the Gaussian that is smaller than the sigma.
        """
        freq_increment = np.mean(np.diff(fit_data.x_data))

        fit_a = fit_data.ufloat_params["a"]
        fit_b = fit_data.ufloat_params["b"]
        fit_freq = fit_data.ufloat_params["freq"]
        fit_sigma = fit_data.ufloat_params["sigma"]

        snr = abs(fit_a.n) / np.sqrt(abs(np.median(fit_data.y_data) - fit_b.n))
        fit_width_ratio = fit_sigma.n / np.ptp(fit_data.x_data)

        criteria = [
            fit_data.x_range[0] <= fit_freq.n <= fit_data.x_range[1],
            1.5 * freq_increment < fit_sigma.n,
            fit_width_ratio < 0.25,
            0 < fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(fit_sigma),
            snr > 2,
        ]

        if all(criteria):
            return "good"

        return "bad"
