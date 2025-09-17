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
            include_confidence_bands: bool = False,
            figsize: tuple = (12, 8)
    ) -> Figure:
        """
        Visualize the percentage of ultimate losses emerged over development periods.
        
        This method creates a sophisticated actuarial visualization showing how claims
        develop from initial reporting to ultimate settlement. It's particularly valuable
        for specialty insurance lines where understanding development velocity is critical
        for reserving decisions and regulatory compliance.
        
        The visualization displays:
        1. Individual accident year development patterns (observed vs projected)
        2. Volume-weighted average development pattern across all years
        3. Clear distinction between actual emergence and projected patterns
        
        Mathematical Foundation:
        - Percentage of Ultimate = Cumulative Losses / Ultimate Losses
        - Ultimate Losses = Latest Observed × Remaining CDF
        - Remaining CDF = CDF[latest_period] (from fitted development patterns)
        
        Actuarial Applications:
        - Reserve adequacy assessment and validation
        - Development pattern analysis and benchmarking
        - Tail development identification for long-tail lines
        - Regulatory reporting and audit documentation
        - Management presentations and stakeholder communication

        Parameters
        ----------
        show_by_accident_year : bool, default=True
            Whether to display individual accident year development curves.
            - True: Shows both observed (solid lines) and projected (dashed lines) 
              development for each accident year
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

        include_confidence_bands : bool, default=False
            Whether to include confidence intervals around the average pattern.
            Note: Currently not implemented but reserved for future enhancement.
            
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

        Notes
        -----
        This method automatically fits a Development object to raw triangles to
        obtain the cumulative development factors (CDFs) needed for ultimate
        projections. For already-fitted Development objects, it uses the
        existing CDF patterns.
        
        The method handles triangular data structure correctly, accounting for
        varying numbers of observed development periods across accident years.
        
        Color Scheme:
        - Light blue: Individual accident year observed development
        - Red/dashed: Individual accident year projected development  
        - Dark blue: Average development pattern
        
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

        # =================================================================
        # STEP 2: PREPARE DEVELOPMENT DATA AND CDF PATTERNS
        # =================================================================
        
        # Import Development class for fitting CDF patterns if needed
        from chainladder.development import Development

        # Determine data source: use existing CDF if available, otherwise fit new patterns
        if hasattr(self, 'cdf_'):
            # CASE A: Already fitted Development object with CDF
            # Extract the cumulative development factors directly
            cdf_values = self.cdf_.compute().set_backend("numpy").values
            if cdf_values.ndim == 4:
                avg_cdf_data = cdf_values[0, 0, 0]
            else:
                avg_cdf_data = cdf_values[0, 0]
            development_periods = self.cdf_.development.copy()
        elif any('Ult' in str(d) for d in self.development):
            # CASE B: CDF Triangle (development periods contain 'Ult')
            # This IS already a CDF triangle, so extract the data directly
            cdf_values = self.compute().set_backend("numpy").values[0, 0]
            # Handle both (n,) and (1, n) shaped arrays
            avg_cdf_data = cdf_values.flatten() if cdf_values.ndim > 1 else cdf_values
            development_periods = self.development.copy()
        else:
            # CASE C: Raw Triangle - need to fit Development patterns
            # Fit a Development object to get CDF patterns to ultimate
            dev_obj = Development().fit(self)
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
        
        # Process individual accident year patterns only for raw triangles
        # (fitted objects and CDF triangles already have averaged patterns and don't need individual processing)
        if show_by_accident_year and not hasattr(self, 'cdf_') and not any('Ult' in str(d) for d in self.development):

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

                # For percentage of ultimate, we need n_observed - 1 values
                # (the last observed period represents "ultimate" for that accident year so far)
                if n_observed <= 1:
                    # Skip accident years with only 1 or 0 observations (no development to show)
                    continue

                observed_pct = []
                for j in range(n_observed - 1):  # Exclude the last period
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
            # For fitted Development objects and CDF triangles, individual year patterns aren't meaningful
            # since the data has already been averaged/smoothed
            individual_percent_ult = None
            show_by_accident_year = False

        # =================================================================
        # STEP 4: CALCULATE AVERAGE DEVELOPMENT PATTERN
        # =================================================================
        
        # Calculate the overall average percentage of ultimate pattern
        # This represents the average development pattern from CDF
        # Note: For average pattern, percentage of ultimate = 1 / CDF (theoretical relationship)
        avg_percent_ult = 1.0 / avg_cdf_data

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
                        # We need the first len(observed_pct) periods
                        relevant_periods = development_periods[:len(observed_pct)]
                        obs_x = relevant_periods[obs_mask]
                        obs_y = observed_pct_array[obs_mask]

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
                x_values = development_periods[mask]
                y_plot = avg_percent_ult[mask]
                ax.plot(x_values, y_plot,
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