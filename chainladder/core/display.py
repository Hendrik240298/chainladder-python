# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import IPython.display
import numpy as np
import pandas as pd
import re

from typing import TYPE_CHECKING

try:
    from IPython.core.display import HTML
except ImportError:
    HTML = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    plt = None
    Figure = None

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        IndexSlice,
        Series
    )
    from matplotlib.figure import Figure

class TriangleDisplay:

    def __repr__(self) -> str | DataFrame:

        # If values hasn't been defined yet, return an empty triangle.
        if self._dimensionality == 'empty':
            return "Empty Triangle."

        # For triangles with a single segment, containing a single triangle, return the
        # DataFrame of the values.
        elif self._dimensionality == 'single':
            data: DataFrame = self._repr_format()
            return data.to_string()

        # For multidimensional triangles, return a summary.
        else:
            return self._summary_frame().__repr__()

    def _summary_frame(self) -> DataFrame:
        """
        Returns summary information about the triangle. Used in the case of multidimensional triangles.

        Returns
        -------

        DataFrame
        """
        return pd.Series(
            data=[
                self.valuation_date.strftime("%Y-%m"),
                "O" + self.origin_grain + "D" + self.development_grain,
                self.shape,
                self.key_labels,
                list(self.vdims),
            ],
            index=["Valuation:", "Grain:", "Shape:", "Index:", "Columns:"],
            name="Triangle Summary",
        ).to_frame()

    def _repr_html_(self) -> str:
        """
        Jupyter/Ipython HTML representation.

        Returns
        -------
        str
        """

        # Case empty triangle.
        if self._dimensionality == 'empty':
            return "Empty Triangle."

        # Case single-dimensional triangle.
        elif self._dimensionality == 'single':
            data = self._repr_format()
            fmt_str = self._get_format_str(data=data)
            default = (
                data.to_html(
                    max_rows=pd.options.display.max_rows,
                    max_cols=pd.options.display.max_columns,
                    float_format=fmt_str.format,
                )
                .replace("nan", "")
                .replace("NaN", "")
            )
            return default
        # Case multidimensional triangle.
        else:
            return self._summary_frame().to_html(
                max_rows=pd.options.display.max_rows,
                max_cols=pd.options.display.max_columns,
            )

    @staticmethod
    def _get_format_str(data: DataFrame) -> str:
        """
        Returns a numerical format string based on the magnitude of the mean absolute value of the values in the
        supplied DataFrame.

        Returns
        -------
        str
        """
        if np.all(np.isnan(data)):
            return ""
        elif np.nanmean(abs(data)) < 10:
            return "{0:,.4f}"
        elif np.nanmean(abs(data)) < 1000:
            return "{0:,.2f}"
        else:
            return "{:,.0f}"

    def _repr_format(
            self,
            origin_as_datetime: bool = False
    ) -> DataFrame:
        """
        Prepare triangle values for printing as a DataFrame. Mainly used with single-dimensional triangles.

        Returns
        -------
        DataFrame
        """
        out: np.ndarray = self.compute().set_backend("numpy").values[0, 0]
        if origin_as_datetime and not self.is_pattern:
            origin: Series = self.origin.to_timestamp(how="s")
        else:
            origin = self.origin.copy()
        origin.name = None

        if self.origin_grain == "S" and not origin_as_datetime:
            origin_formatted = [""] * len(origin)
            for origin_index in range(len(origin)):
                origin_formatted[origin_index] = (
                    origin.astype("str")[origin_index]
                    .replace("Q1", "H1")
                    .replace("Q3", "H2")
                )
            origin = origin_formatted
        development = self.development.copy()
        development.name = None
        return pd.DataFrame(out, index=origin, columns=development)

    def heatmap(
            self,
            cmap: str = "coolwarm",
            low: float = 0,
            high: float = 0,
            axis: int | str = 0,
            subset: IndexSlice=None
    ) -> IPython.display.HTML:
        """
        Color the background in a gradient according to the data in each
        column (optionally row). Requires matplotlib.

        Parameters
        ----------

        cmap : str or colormap
            matplotlib colormap
        low, high : float
            compress the range by these values.
        axis : int or str
            The axis along which to apply heatmap
        subset : IndexSlice
            a valid slice for data to limit the style application to

        Returns
        -------
            Ipython.display.HTML

        """
        if self._dimensionality == 'single':
            data = self._repr_format()
            fmt_str = self._get_format_str(data)

            axis = self._get_axis(axis)

            raw_rank = data.rank(axis=axis)
            shape_size = data.shape[axis]
            rank_size = data.rank(axis=axis).max(axis=axis)
            gmap = (raw_rank - 1).div(rank_size - 1, axis=not axis) * (
                shape_size - 1
            ) + 1
            gmap = gmap.replace(np.nan, (shape_size + 1) / 2)
            if pd.__version__ >= "1.3":
                default_output = (
                    data.style.format(fmt_str)
                    .background_gradient(
                        cmap=cmap,
                        low=low,
                        high=high,
                        axis=None,
                        subset=subset,
                        gmap=gmap,
                    )
                    .to_html()
                )
            else:
                default_output = (
                    data.style.format(fmt_str)
                    .background_gradient(
                        cmap=cmap,
                        low=low,
                        high=high,
                        axis=axis,
                    )
                    .render()
                )
            output_xnan = re.sub("<td.*nan.*td>", "<td></td>", default_output)
        else:
            raise ValueError("heatmap() only works with a single triangle.")
        if HTML:
            return HTML(output_xnan)
        elif HTML is None:
            raise ImportError("heatmap requires IPython.")

    def percent_of_ultimate(
            self,
            show_by_origin: bool = True,
            show_average_pattern: bool = True,
            show_origin_years_in_legend: bool = True,
            selected_origins: list = None,
            average: str | float = "volume",
            figsize: tuple = (12, 8)
    ) -> Figure:
        """
        Visualize development patterns as percentage of ultimate.

        Shows individual origin development curves and volume-weighted average
        pattern to analyze emergence velocity and volatility.

        Parameters
        ----------
        show_by_origin : bool, default=True
            Display individual origin development curves
        show_average_pattern : bool, default=True
            Display volume-weighted average development pattern
        show_origin_years_in_legend : bool, default=True
            Show origin years in legend instead of generic labels
        selected_origins : list, optional
            Specific origins to plot. If None, plots all origins
        average : str or float, default="volume"
            Averaging method: "volume", "simple", "regression", or numeric
        figsize : tuple, default=(12, 8)
            Figure size in inches

        Returns
        -------
        matplotlib.figure.Figure
            Percentage of ultimate visualization
        """
        # Validate inputs
        if plt is None:
            raise ImportError("percent_of_ultimate requires matplotlib.")

        # Check single triangle
        if self._dimensionality != 'single':
            raise ValueError("percent_of_ultimate() only works with a single triangle.")

        # Check raw triangle
        if hasattr(self, 'cdf_'):
            raise ValueError(
                "percent_of_ultimate() requires raw triangle data for individual origin analysis. "
                "Use the original triangle before fitting Development patterns. "
                "For fitted objects, use: original_triangle.percent_of_ultimate(average='volume')"
            )

        if any('Ult' in str(d) for d in self.development):
            raise ValueError(
                "percent_of_ultimate() requires raw triangle data, not CDF triangles. "
                "CDF triangles don't contain individual origin information needed for this visualization."
            )

        # Validate average
        valid_string_averages = ["volume", "simple", "regression"]
        if isinstance(average, str):
            if average not in valid_string_averages:
                raise ValueError(f"average must be one of {valid_string_averages}")
        elif isinstance(average, (int, float)):
            # Numeric values not fully supported yet
            import warnings
            warnings.warn(f"Numeric average values ({average}) are not fully supported yet. Using 'volume' averaging instead.")
            average = "volume"
        else:
            raise ValueError(f"average must be a string from {valid_string_averages}, got {type(average)}")

        # Prepare development data
        from chainladder.development import Development
        dev_obj = Development(average=average).fit(self)
        avg_cdf_data = dev_obj.cdf_.compute().set_backend("numpy").values[0, 0, 0]
        development_periods = dev_obj.cdf_.development.copy()

        # Get triangle data once for all calculations
        triangle_values = self.compute().set_backend("numpy").values[0, 0]

        # Process selected origins (inline)
        if selected_origins is None:
            selected_origin_indices = list(range(len(self.origin)))
        else:
            import warnings
            available_origins = [str(origin) for origin in self.origin]
            selected_origin_indices = [
                available_origins.index(str(origin))
                for origin in selected_origins
                if str(origin) in available_origins
            ]
            if not selected_origin_indices:
                warnings.warn("No valid origins found in selection. Using all origins.")
                selected_origin_indices = list(range(len(self.origin)))

        # Calculate ultimate projections once for reuse
        ultimate_projections = []
        for i in range(triangle_values.shape[0]):
            observed_cumulative = triangle_values[i, :]
            n_observed = (~np.isnan(observed_cumulative)).sum()
            if n_observed > 0:
                latest_observed = observed_cumulative[~np.isnan(observed_cumulative)][-1]
                latest_period_idx = n_observed - 1
                if latest_period_idx < len(avg_cdf_data):
                    ultimate_projected = latest_observed * avg_cdf_data[latest_period_idx]
                else:
                    ultimate_projected = latest_observed
            else:
                ultimate_projected = np.nan
            ultimate_projections.append(ultimate_projected)

        # Calculate individual origin patterns
        if show_by_origin:
            individual_data = []
            for i in range(triangle_values.shape[0]):
                observed_cumulative = triangle_values[i, :]
                n_observed = (~np.isnan(observed_cumulative)).sum()
                ultimate_projected = ultimate_projections[i]

                if n_observed <= 1 or np.isnan(ultimate_projected):
                    continue

                # Check if origin is complete
                is_complete = (n_observed == triangle_values.shape[1] and
                              not np.isnan(observed_cumulative[-1]))
                end_period = n_observed if is_complete else n_observed - 1

                observed_pct = [
                    observed_cumulative[j] / ultimate_projected if not np.isnan(observed_cumulative[j]) else np.nan
                    for j in range(end_period)
                ]
                individual_data.append((observed_pct, n_observed))

            individual_percent_ult = individual_data
        else:
            individual_percent_ult = None

        # Calculate average development pattern
        max_dev_periods = triangle_values.shape[1]
        weighted_sums = np.zeros(max_dev_periods)
        total_weights = np.zeros(max_dev_periods)

        # Define weight function based on averaging method
        def get_weight(ultimate_proj, cumulative_val):
            if average == "volume":
                return ultimate_proj
            elif average in ["simple", "regression"]:
                return 1.0
            elif isinstance(average, (int, float)):
                exponent = 2 - average
                base_weight = cumulative_val if cumulative_val > 0 else 1.0
                return base_weight ** (1 - exponent) if exponent != 1 else base_weight
            else:
                return ultimate_proj

        for i in range(triangle_values.shape[0]):
            observed_cumulative = triangle_values[i, :]
            n_observed = (~np.isnan(observed_cumulative)).sum()
            ultimate_projected = ultimate_projections[i]

            if n_observed <= 1 or np.isnan(ultimate_projected):
                continue

            # Check if origin is complete
            is_complete = (n_observed == triangle_values.shape[1] and
                          not np.isnan(observed_cumulative[-1]))
            end_period = n_observed if is_complete else n_observed - 1

            for j in range(end_period):
                if not np.isnan(observed_cumulative[j]) and j < max_dev_periods:
                    pct_ultimate = observed_cumulative[j] / ultimate_projected
                    weight = get_weight(ultimate_projected, observed_cumulative[j])
                    weighted_sums[j] += pct_ultimate * weight
                    total_weights[j] += weight

        avg_percent_ult = np.full(max_dev_periods, np.nan)
        mask = total_weights > 0
        avg_percent_ult[mask] = weighted_sums[mask] / total_weights[mask]

        # Create visualization
        
        fig, ax = plt.subplots(figsize=figsize)

        # Plot individual patterns
        if show_by_origin and individual_percent_ult is not None:
            # Prepare labels for all selected origins
            if show_origin_years_in_legend or selected_origins is not None:
                labels = [str(self.origin[i]) for i in selected_origin_indices]
            else:
                labels = ['Individual origin'] + [''] * (len(selected_origin_indices) - 1)

            label_idx = 0
            for i, (observed_pct, n_observed) in enumerate(individual_percent_ult):
                if i not in selected_origin_indices:
                    continue

                if len(observed_pct) > 0:
                    observed_pct_array = np.array(observed_pct)
                    obs_mask = ~np.isnan(observed_pct_array)

                    if obs_mask.any():
                        n_available_periods = min(len(observed_pct), len(development_periods))
                        obs_x = development_periods[:n_available_periods][obs_mask[:n_available_periods]]
                        obs_y = observed_pct_array[:n_available_periods][obs_mask[:n_available_periods]]

                        # Add ultimate period if available
                        if len(observed_pct) > len(development_periods):
                            ultimate_idx = len(development_periods)
                            if ultimate_idx < len(observed_pct) and not np.isnan(observed_pct[ultimate_idx]):
                                last_dev_period = development_periods.iloc[-1]
                                ultimate_label = "Ult" if isinstance(last_dev_period, str) else f"{last_dev_period + 12}-Ult"
                                obs_x = list(obs_x) + [ultimate_label]
                                obs_y = list(obs_y) + [observed_pct[ultimate_idx]]

                        ax.plot(obs_x, obs_y, alpha=0.6, linewidth=1.5, linestyle='-',
                               marker='o', markersize=4, label=labels[label_idx])
                        label_idx += 1

        # Plot average pattern
        if show_average_pattern:
            mask = ~np.isnan(avg_percent_ult)
            if mask.any():
                n_plot_periods = min(len(development_periods), len(avg_percent_ult))

                # Build x and y arrays
                x_plot = [development_periods.iloc[i] for i in range(n_plot_periods)
                         if not np.isnan(avg_percent_ult[i])]
                y_plot = [avg_percent_ult[i] for i in range(n_plot_periods)
                         if not np.isnan(avg_percent_ult[i])]

                # Add ultimate period if available
                if len(avg_percent_ult) > len(development_periods):
                    ultimate_idx = len(development_periods)
                    if ultimate_idx < len(avg_percent_ult) and not np.isnan(avg_percent_ult[ultimate_idx]):
                        last_dev_period = development_periods.iloc[-1]
                        ultimate_label = "Ult" if isinstance(last_dev_period, str) else f"{last_dev_period + 12}-Ult"
                        x_plot.append(ultimate_label)
                        y_plot.append(avg_percent_ult[ultimate_idx])

                if x_plot and y_plot:
                    ax.plot(x_plot, y_plot, color='darkblue', linewidth=3,
                           marker='o', markersize=6, label='Average Pattern')

        # Format plot
        ax.set_xlabel('Development Period')
        ax.set_ylabel('% of Ultimate')
        ax.set_title('Development Pattern: Percentage of Ultimate')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        plt.xticks(rotation=45)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        return fig

    @property
    def _dimensionality(self) -> str:
        """
        Determine whether the triangle is empty, single-dimensional, or multidimensional. Used for conditional
        branching in displaying the triangle.

        Returns
        -------
        str
        """
        try:
             self.values
        except AttributeError:
            return 'empty'

        if (self.values.shape[0], self.values.shape[1]) == (1, 1):
            return 'single'

        else :
            return 'multi'