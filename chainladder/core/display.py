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
            show_by_accident_year: bool = True,
            show_average_pattern: bool = True,
            show_origin_years_in_legend: bool = True,
            selected_origins: list = None,
            average: str | float = "volume",
            figsize: tuple = (12, 8)
    ) -> Figure:
        """
        Visualize individual accident year development patterns as percentage of ultimate.

        This method analyzes raw triangle data to show actual emergence patterns
        for each accident year, providing insights into development velocity and
        volatility essential for specialty insurance reserving.

        The visualization displays:
        1. Individual accident year observed development patterns
        2. Volume-weighted average development pattern across all years
        3. Clear view of actual emergence vs. ultimate projections

        Mathematical Foundation:
        - Percentage of Ultimate = Cumulative Losses / Ultimate Losses
        - Ultimate Losses = Latest Observed × CDF_to_ultimate
        - CDF patterns fitted using specified averaging method

        Actuarial Applications:
        - Reserve adequacy assessment and validation
        - Development pattern analysis and benchmarking
        - Individual year volatility assessment
        - Regulatory reporting and audit documentation
        - Management presentations and stakeholder communication

        IMPORTANT: Only works with raw triangles to ensure individual accident
        year patterns can be calculated from actual observed data.

        Parameters
        ----------
        show_by_accident_year : bool, default=True
            Whether to display individual accident year development curves.
            - True: Shows observed development patterns for each accident year
            - False: Displays only the average pattern for cleaner visualization
            
        show_average_pattern : bool, default=True
            Whether to display the volume-weighted average development pattern.
            - True: Shows the overall pattern as a prominent line with markers
            - False: Hides the average pattern to focus on individual variations

        show_origin_years_in_legend : bool, default=True
            Whether to show individual accident years in the legend.
            - True: Each accident year appears as a separate legend entry (e.g., "1981", "1982")
            - False: Shows only generic labels ("Individual origin", "Average Pattern")

        selected_origins : list of int or str, optional
            Specific accident years to plot. If None (default), plots all available origins.
            Can accept years as integers (1995, 1997) or strings ("1995", "1997").
            Useful for focusing analysis on specific periods or reducing visual clutter.
            Example: [1995, 1997, 2005, 2003]

        average : str or float, default="volume"
            Method for averaging development factors when fitting patterns to raw triangles.
            Only applies to raw triangles; ignored for CDF/fitted triangles.

            String options:
            - "volume": Volume-weighted averaging (default)
            - "simple": Simple arithmetic averaging
            - "regression": Unweighted regression

            Numeric options (Zehnwirth & Barnett style):
            - Float values control the weighting exponent as (2-average)
            - Examples: 1.5 → exponent=0.5, 0.8 → exponent=1.2
            - Allows fine-tuning between standard averaging methods

        figsize : tuple of float, default=(12, 8)
            Figure dimensions in inches as (width, height).
            Recommended sizes:
            - (12, 8): Standard presentation format
            - (14, 10): Detailed analysis format
            - (10, 6): Compact report format

        Returns
        -------
        matplotlib.figure.Figure
            A matplotlib Figure object containing the percentage of ultimate visualization.
            The figure includes:
            - Individual accident year patterns (if enabled)
            - Average development pattern (if enabled)
            - Professional styling suitable for actuarial reports
            - Legend distinguishing between observed and projected data
            - Grid lines for easier value reading

        Raises
        ------
        ImportError
            If matplotlib is not installed and available for plotting.
            
        ValueError
            If the triangle is multidimensional. This method only works with
            single triangles (shape[0] == 1 and shape[1] == 1).
            If the triangle is a fitted Development object or CDF triangle.
            This method requires raw triangle data for individual pattern analysis.

        Notes
        -----
        This method fits a Development object to the raw triangle using the
        specified averaging method to obtain cumulative development factors (CDFs)
        for ultimate projections.

        The method correctly handles triangular data structure, accounting for
        varying numbers of observed development periods across accident years.

        Color Scheme:
        - Individual accident years: Colored lines with markers
        - Dark blue: Volume-weighted average development pattern
        
        Examples
        --------
        >>> import chainladder as cl
        >>> raa = cl.load_sample('raa')
        >>> 
        >>> # Basic usage showing all patterns
        >>> fig = raa.percent_of_ultimate()
        >>> 
        >>> # Focus on average pattern only
        >>> fig = raa.percent_of_ultimate(
        ...     show_by_accident_year=False,
        ...     show_average_pattern=True
        ... )
        >>> 
        >>> # Show individual accident years in legend
        >>> fig = raa.percent_of_ultimate(show_origin_years_in_legend=True)
        >>>
        >>> # Plot specific accident years only
        >>> fig = raa.percent_of_ultimate(selected_origins=[1985, 1987, 1990])
        >>>
        >>> # Custom size for reports
        >>> fig = raa.percent_of_ultimate(figsize=(14, 10))
        >>>
        >>> # Different averaging methods (raw triangles only)
        >>> fig = raa.percent_of_ultimate(average="simple")
        >>> fig = raa.percent_of_ultimate(average="regression")
        >>> fig = raa.percent_of_ultimate(average=1.5)  # Zehnwirth & Barnett style
        
        See Also
        --------
        Triangle.link_ratio : Calculate age-to-age development factors
        Development.fit : Fit development patterns to triangle data
        Triangle.ldf_ : Access fitted link development factors
        """
        # =================================================================
        # STEP 1: INPUT VALIDATION AND DEPENDENCY CHECKS
        # =================================================================

        # Ensure matplotlib is available for plotting functionality
        if plt is None:
            raise ImportError("percent_of_ultimate requires matplotlib.")

        # Validate that we're working with a single triangle (not multidimensional)
        # Multidimensional triangles would require aggregation logic not yet implemented
        if self._dimensionality != 'single':
            raise ValueError("percent_of_ultimate() only works with a single triangle.")

        # Validate that input is a raw triangle (not fitted/CDF)
        if hasattr(self, 'cdf_'):
            raise ValueError(
                "percent_of_ultimate() requires raw triangle data for individual accident year analysis. "
                "Use the original triangle before fitting Development patterns. "
                "For fitted objects, use: original_triangle.percent_of_ultimate(average='volume')"
            )

        if any('Ult' in str(d) for d in self.development):
            raise ValueError(
                "percent_of_ultimate() requires raw triangle data, not CDF triangles. "
                "CDF triangles don't contain individual accident year information needed for this visualization."
            )

        # Validate average parameter
        valid_string_averages = ["volume", "simple", "regression"]
        if isinstance(average, str):
            if average not in valid_string_averages:
                raise ValueError(f"average must be one of {valid_string_averages}")
        elif isinstance(average, (int, float)):
            # Numeric Zehnwirth & Barnett values are not yet fully supported by the Development class
            # For now, fallback to volume weighting with a warning
            import warnings
            warnings.warn(f"Numeric average values ({average}) are not fully supported yet. Using 'volume' averaging instead.")
            average = "volume"
        else:
            raise ValueError(f"average must be a string from {valid_string_averages}, got {type(average)}")

        # =================================================================
        # STEP 2: PREPARE DEVELOPMENT DATA AND CDF PATTERNS
        # =================================================================

        # Import Development class for fitting CDF patterns
        from chainladder.development import Development

        # Fit Development object to raw triangle using specified averaging method
        dev_obj = Development(average=average).fit(self)
        avg_cdf_data = dev_obj.cdf_.compute().set_backend("numpy").values[0, 0, 0]
        development_periods = dev_obj.cdf_.development.copy()

        # =================================================================
        # STEP 2.5: VALIDATE AND PROCESS SELECTED ORIGINS
        # =================================================================

        def _validate_and_process_origins(selected_origins):
            """Validate and convert selected origins to indices"""
            if selected_origins is None:
                return list(range(len(self.origin)))  # All origins

            import warnings
            available_origins = [str(origin) for origin in self.origin]
            selected_indices = []

            for origin in selected_origins:
                origin_str = str(origin)
                if origin_str in available_origins:
                    selected_indices.append(available_origins.index(origin_str))
                else:
                    warnings.warn(f"Origin {origin} not found in triangle data. Available origins: {available_origins}")

            if not selected_indices:
                warnings.warn("No valid origins found in selection. Using all origins.")
                return list(range(len(self.origin)))

            return selected_indices

        selected_origin_indices = _validate_and_process_origins(selected_origins)

        # =================================================================
        # STEP 3: CALCULATE INDIVIDUAL ACCIDENT YEAR PATTERNS
        # =================================================================
        
        # Process individual accident year patterns
        # (all inputs are now raw triangles after validation)
        if show_by_accident_year:

            # Extract the triangle data as numpy array for efficient processing
            triangle_values = self.compute().set_backend("numpy").values[0, 0]  # shape: (n_origins, n_dev_periods)

            # Initialize storage for individual accident year data
            # Each element will be a tuple: (observed_percentages, projected_percentages, n_observed_periods)
            individual_data = []

            # =================================================================
            # STEP 3A: PROCESS EACH ACCIDENT YEAR INDIVIDUALLY
            # =================================================================

            for i in range(triangle_values.shape[0]):
                # Extract observed cumulative losses for this accident year
                observed_cumulative = triangle_values[i, :]

                # Count how many development periods have actual (non-NaN) observations
                n_observed = (~np.isnan(observed_cumulative)).sum()

                # Skip accident years with no observed data
                if n_observed == 0:
                    continue

                # =================================================================
                # STEP 3B: CALCULATE ULTIMATE PROJECTION FOR THIS ACCIDENT YEAR
                # =================================================================

                # Get the latest observed cumulative loss value
                latest_observed = observed_cumulative[~np.isnan(observed_cumulative)][-1]

                # Calculate the index of the latest observed period (0-based)
                latest_period_idx = n_observed - 1

                # =================================================================
                # STEP 3C: PROJECT ULTIMATE LOSSES USING AVERAGE CDF PATTERN
                # =================================================================

                # Apply the CDF from the latest observed period to ultimate
                # CDF represents the factor to go from current period to ultimate
                if latest_period_idx < len(avg_cdf_data):
                    cdf_to_ultimate = avg_cdf_data[latest_period_idx]
                    ultimate_projected = latest_observed * cdf_to_ultimate
                else:
                    # If beyond available CDF data, assume no further development
                    ultimate_projected = latest_observed
                
                # =================================================================
                # STEP 3D: CALCULATE PERCENTAGE OF ULTIMATE FOR OBSERVED PERIODS
                # =================================================================
                
                # Calculate percentage of ultimate for observed development periods only
                # Each accident year should have a different number of percentages based on its development
                # Formula: % of Ultimate = Cumulative Loss / Ultimate Loss

                # Calculate percentage of ultimate for observed periods
                # For complete accident years (those with ultimate values), include the ultimate period (100%)
                # For incomplete years, exclude the last period as it's not yet ultimate
                if n_observed <= 1:
                    # Skip accident years with only 1 or 0 observations (no development to show)
                    continue

                observed_pct = []

                # Check if this accident year is complete (has reached ultimate)
                is_complete = (n_observed == triangle_values.shape[1] and
                              not np.isnan(observed_cumulative[-1]))

                # For complete years, include all periods including ultimate (which will be 100%)
                # For incomplete years, exclude the last period
                end_period = n_observed if is_complete else n_observed - 1

                for j in range(end_period):
                    if not np.isnan(observed_cumulative[j]):
                        observed_pct.append(observed_cumulative[j] / ultimate_projected)
                    else:
                        observed_pct.append(np.nan)

                # =================================================================
                # STEP 3E: STORE RESULTS FOR THIS ACCIDENT YEAR
                # =================================================================

                # Store as tuple: (observed percentages, number of observed periods)
                # Note: No more projections stored
                individual_data.append((observed_pct, n_observed))
                
            # Store the processed individual accident year data
            individual_percent_ult = individual_data
        else:
            # No individual patterns requested
            individual_percent_ult = None

        # =================================================================
        # STEP 4: CALCULATE AVERAGE DEVELOPMENT PATTERN
        # =================================================================

        # Calculate volume-weighted average percentage of ultimate pattern from raw triangle data
        # This is consistent with individual accident year calculations
        triangle_values = self.compute().set_backend("numpy").values[0, 0]

        # Initialize arrays to store weighted sums and total weights for each development period
        # Include ultimate period to allow for 100% values in complete accident years
        max_dev_periods = triangle_values.shape[1]  # Include ultimate period
        weighted_sums = np.zeros(max_dev_periods)
        total_weights = np.zeros(max_dev_periods)

        for i in range(triangle_values.shape[0]):
            observed_cumulative = triangle_values[i, :]
            n_observed = (~np.isnan(observed_cumulative)).sum()

            if n_observed <= 1:
                continue

            # Calculate ultimate for this accident year (same logic as individual patterns)
            latest_observed = observed_cumulative[~np.isnan(observed_cumulative)][-1]
            latest_period_idx = n_observed - 1

            if latest_period_idx < len(avg_cdf_data):
                cdf_to_ultimate = avg_cdf_data[latest_period_idx]
                ultimate_projected = latest_observed * cdf_to_ultimate
            else:
                ultimate_projected = latest_observed

            # Calculate percentage of ultimate for each observed period
            # Include ultimate period for complete accident years, exclude for incomplete ones
            is_complete = (n_observed == triangle_values.shape[1] and
                          not np.isnan(observed_cumulative[-1]))

            # For complete years, include all periods including ultimate (which will be 100%)
            # For incomplete years, exclude the last period
            end_period = n_observed if is_complete else n_observed - 1

            for j in range(end_period):
                if not np.isnan(observed_cumulative[j]) and j < max_dev_periods:
                    pct_ultimate = observed_cumulative[j] / ultimate_projected

                    # Apply weighting consistent with the chosen averaging method
                    if average == "volume":
                        weight = ultimate_projected  # Volume weighting
                    elif average == "simple":
                        weight = 1.0  # Simple (equal) weighting
                    elif average == "regression":
                        weight = 1.0  # Unweighted (same as simple for this calculation)
                    else:  # Zehnwirth & Barnett numeric style
                        # For numeric values, use a simplified weighting approach
                        # This approximates the Development class behavior in the context of percentage calculation
                        if isinstance(average, (int, float)):
                            exponent = 2 - average
                            # Use observed cumulative as base for weighting (similar to Development class logic)
                            base_weight = observed_cumulative[j] if observed_cumulative[j] > 0 else 1.0
                            weight = base_weight ** (1 - exponent) if exponent != 1 else base_weight
                        else:
                            weight = ultimate_projected  # Fallback to volume weighting

                    weighted_sums[j] += pct_ultimate * weight
                    total_weights[j] += weight

        # Calculate weighted averages using the specified averaging method
        avg_percent_ult = np.full(max_dev_periods, np.nan)
        mask = total_weights > 0
        avg_percent_ult[mask] = weighted_sums[mask] / total_weights[mask]

        # =================================================================
        # STEP 5: CREATE MATPLOTLIB VISUALIZATION
        # =================================================================
        
        # Initialize the figure with specified dimensions
        fig, ax = plt.subplots(figsize=figsize)

        # =================================================================
        # STEP 5A: PLOT INDIVIDUAL ACCIDENT YEAR PATTERNS
        # =================================================================
        
        # Only plot individual patterns for raw triangles (not fitted objects)
        if show_by_accident_year and individual_percent_ult is not None:

            for i, (observed_pct, n_observed) in enumerate(individual_percent_ult):

                # Skip this origin if not in selected origins
                if i not in selected_origin_indices:
                    continue

                # ---------------------------------------------------------------
                # Plot Observed Development (Solid Lines Only)
                # ---------------------------------------------------------------
                if len(observed_pct) > 0:
                    # Convert to numpy array for easier handling
                    observed_pct_array = np.array(observed_pct)
                    obs_mask = ~np.isnan(observed_pct_array)

                    if obs_mask.any():
                        # Get the correct development periods for this accident year
                        # Handle case where observed_pct might be longer than development_periods
                        n_available_periods = min(len(observed_pct), len(development_periods))
                        relevant_periods = development_periods[:n_available_periods]

                        # Adjust mask to match available periods
                        obs_mask_adjusted = obs_mask[:n_available_periods]
                        obs_x = relevant_periods[obs_mask_adjusted]
                        obs_y = observed_pct_array[:n_available_periods][obs_mask_adjusted]

                        # If there's an ultimate period beyond development_periods, handle it separately
                        if len(observed_pct) > len(development_periods) and len(observed_pct) > n_available_periods:
                            ultimate_idx = len(development_periods)
                            if ultimate_idx < len(observed_pct) and not np.isnan(observed_pct[ultimate_idx]):
                                # Add ultimate point
                                last_dev_period = development_periods.iloc[-1]
                                if isinstance(last_dev_period, str):
                                    ultimate_label = "Ult"
                                else:
                                    ultimate_label = f"{last_dev_period + 12}-Ult"
                                # Extend arrays with ultimate point
                                obs_x = list(obs_x) + [ultimate_label]
                                obs_y = list(obs_y) + [observed_pct[ultimate_idx]]

                        # Determine label for this accident year
                        if show_origin_years_in_legend or selected_origins is not None:
                            # Show the actual accident year in the legend
                            # (always show origin years when specific origins are selected)
                            origin_year = str(self.origin[i])
                            label = origin_year
                        else:
                            # Show generic label only for the first selected accident year
                            first_selected_index = min(selected_origin_indices)
                            label = 'Individual origin' if i == first_selected_index else ""

                        # Plot observed development with professional styling
                        ax.plot(obs_x, obs_y,
                               alpha=0.6,              # Semi-transparent for layering
                               linewidth=1.5,          # Medium line weight
                               linestyle='-',          # Solid line for observed data
                               marker='o',
                               markersize=4,
                               label=label)

        # Plot average pattern
        if show_average_pattern:
            mask = ~np.isnan(avg_percent_ult)
            if mask.any():
                # Handle potential mismatch between development_periods and avg_percent_ult length
                # This can happen when we include ultimate period for complete accident years
                n_plot_periods = min(len(development_periods), len(avg_percent_ult))

                # Create x and y arrays for plotting, handling length mismatch
                x_plot = []
                y_plot = []

                # Add regular development periods
                for i in range(n_plot_periods):
                    if not np.isnan(avg_percent_ult[i]):
                        x_plot.append(development_periods.iloc[i])
                        y_plot.append(avg_percent_ult[i])

                # Add ultimate period if it exists and has data
                if len(avg_percent_ult) > len(development_periods):
                    ultimate_idx = len(development_periods)
                    if ultimate_idx < len(avg_percent_ult) and not np.isnan(avg_percent_ult[ultimate_idx]):
                        # Create ultimate label
                        last_dev_period = development_periods.iloc[-1]
                        if isinstance(last_dev_period, str):
                            ultimate_label = "Ult"
                        else:
                            ultimate_label = f"{last_dev_period + 12}-Ult"

                        x_plot.append(ultimate_label)
                        y_plot.append(avg_percent_ult[ultimate_idx])

                # Plot if we have data
                if x_plot and y_plot:
                    ax.plot(x_plot, y_plot,
                           color='darkblue',
                           linewidth=3,
                           marker='o',
                           markersize=6,
                           label='Average Pattern')

        # Tail development highlighting removed - not needed for this visualization

        # Formatting
        ax.set_xlabel('Development Period')
        ax.set_ylabel('% of Ultimate')
        ax.set_title('Development Pattern: Percentage of Ultimate')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        # Rotate x-axis labels to prevent crowding with many development periods
        plt.xticks(rotation=45)

        # Set y-axis to 0-1 scale without percentage formatting
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