# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from ....logging_manager import LoggerBase


class LayoutOptimization(LoggerBase):
    def __init__(self, fi, boundaries, min_dist=None, freq=None):
        self.fi = fi.copy()

        # Allow boundaries to be set either as a list of corners or as a 
        # nested list of corners (for seperable regions)
        self.boundaries = boundaries
        b_depth = list_depth(boundaries)

        boundary_specification_error_msg = (
            "boundaries should be a list of coordinates (specifed as (x,y) "+\
            "tuples) or as a list of list of tuples (for seperable regions)."
        )
        
        if b_depth == 1:
            self._boundary_polygon = MultiPolygon([Polygon(self.boundaries)])
            self._boundary_line = self._boundary_polygon.boundary
        elif b_depth == 2:
            if not isinstance(self.boundaries[0][0], tuple):
                raise TypeError(boundary_specification_error_msg)
            self._boundary_polygon = MultiPolygon([Polygon(p) for p in self.boundaries])
            self._boundary_line = self._boundary_polygon.boundary
        else:
            raise TypeError(boundary_specification_error_msg) 

        self.xmin, self.ymin, self.xmax, self.ymax = self._boundary_polygon.bounds

        # If no minimum distance is provided, assume a value of 2 rotor diamters
        if min_dist is None:
            self.min_dist = 2 * self.rotor_diameter
        else:
            self.min_dist = min_dist

        # If freq is not provided, give equal weight to all wind conditions
        if freq is None:
            self.freq = np.ones((
                self.fi.floris.flow_field.n_wind_directions,
                self.fi.floris.flow_field.n_wind_speeds
            ))
            self.freq = self.freq / self.freq.sum()
        else:
            self.freq = freq

        self.initial_AEP = fi.get_farm_AEP(self.freq)

    def __str__(self):
        return "layout"

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    # Public methods

    def optimize(self):
        sol = self._optimize()
        return sol

    def plot_layout_opt_results(self):
        x_initial, y_initial, x_opt, y_opt = self._get_initial_and_final_locs()

        plt.figure(figsize=(9, 6))
        fontsize = 16
        plt.plot(x_initial, y_initial, "ob")
        plt.plot(x_opt, y_opt, "or")
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel("x (m)", fontsize=fontsize)
        plt.ylabel("y (m)", fontsize=fontsize)
        plt.axis("equal")
        plt.grid()
        plt.tick_params(which="both", labelsize=fontsize)
        plt.legend(
            ["Old locations", "New locations"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )

        for line in self._boundary_line.geoms:
            xy = np.array(line.coords)
            plt.plot(xy[:,0], xy[:,1], color="b")

        plt.show()

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = self.fi.floris.farm.n_turbines
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fi.floris.farm.rotor_diameters_sorted[0][0][0]

# Helper functions

def list_depth(l):
    if isinstance(l, list) and len(l) > 0:
        return 1 + max(list_depth(item) for item in l)
    else:
        return 0