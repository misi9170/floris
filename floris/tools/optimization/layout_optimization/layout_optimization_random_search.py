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

#TODO LIST
# INCLUDE HET MAPS (Actually I think happens automatically)
# INCLUDE LOGGING
# Allow multiple zones
# Update probability distribution handling
# STRETCH: allow parallelization within a node for flow evaluations,
# similar to how the ParallelInterface handles wake steering optimization
# (possibly future work)



from numpy import random
from multiprocessing import Pool
from time import perf_counter as timerpc

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from shapely.geometry import Point, Polygon

from floris.tools.uncertainty_interface import FlorisInterface

from .layout_optimization_base import LayoutOptimization

from floris.tools.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)

def _load_local_floris_object(
    fi_dict,
):
    # Load local FLORIS object
    fi = FlorisInterface(fi_dict)
    return fi

def test_min_dist(layout_x, layout_y, min_dist):
    coords = np.array([layout_x,layout_y]).T
    dist = pdist(coords)
    return dist.min() >= min_dist

def test_point_in_bounds(test_x, test_y, poly_outer):
    return poly_outer.contains(Point(test_x, test_y))

# Return in MW
def _get_aep(layout_x, layout_y, fi, freq, yaw_angles=None):
    fi.reinitialize(
        layout_x = layout_x,
        layout_y = layout_y
    )

    return fi.get_farm_AEP(freq, yaw_angles=yaw_angles)#/1E6

def _gen_dist_based_init(
    N, # Number of turbins to place
    step_size, #m, courseness of search grid
    poly_outer, # Polygon of outer boundary
    min_x,
    max_x,
    min_y,
    max_y,
    s
):
    """
    Generates an initial layout by randomly placing
    the first turbine than placing the remaining turbines
    as far as possible from the existing turbines.
    """

    # Set random seed
    np.random.seed(s)

    # Choose the initial point randomly
    init_x = float(random.randint(int(min_x),int(max_x)))
    init_y = float(random.randint(int(min_y),int(max_y)))
    while not (poly_outer.contains(Point([init_x,init_y]))):
        init_x = float(random.randint(int(min_x),int(max_x)))
        init_y = float(random.randint(int(min_y),int(max_y)))

    # Intialize the layout arrays
    layout_x = np.array([init_x])
    layout_y = np.array([init_y])
    layout = np.array([layout_x, layout_y]).T

    # Now add the remaining points
    for i in range(1,N):

        print("Placing turbine {0} of {1}.".format(i, N))
        # Add a new turbine being as far as possible from current
        max_dist = 0.
        for x in np.arange(min_x, max_x,step_size):
            for y in np.arange(min_y, max_y,step_size):
                if poly_outer.contains(Point([x,y])):
                    test_dist = cdist([[x,y]],layout)
                    min_dist = np.min(test_dist)
                    if min_dist > max_dist:
                        max_dist = min_dist
                        save_x = x
                        save_y = y

        # Add point to the layout
        layout_x = np.append(layout_x,[save_x])
        layout_y = np.append(layout_y,[save_y])
        layout = np.array([layout_x, layout_y]).T

    # Return the layout
    return layout_x, layout_y

class LayoutOptimizationRandomSearch(LayoutOptimization):
    def __init__(
        self,
        fi,
        boundaries,
        min_dist=None,
        freq=None,
        min_dist_D=None,
        distance_pmf=None,
        n_individuals=4,
        seconds_per_iteration=60.,
        total_optimization_seconds = 600.,
        interface="multiprocessing",  # Options are 'multiprocessing', 'mpi4py', None
        max_workers=None,
        grid_step_size = 100.,
        relegation_number = 1,
        enable_geometric_yaw=False,
        use_dist_based_init=True,
        random_seed=None
    ):
        """
        _summary_

        Args:
            fi (_type_): _description_
            boundaries (iterable(float, float)): Pairs of x- and y-coordinates
                that represent the boundary's vertices (m).
            min_dist (float, optional): The minimum distance to be maintained
                between turbines during the optimization (m). If not specified,
                initializes to 2 rotor diameters. Defaults to None.
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values. If None, equal weight is given to each pair of wind conditions
                Defaults to None.
            min_dist_D (float, optional): The minimum distance to be maintained
                between turbines during the optimization, specified as a multiple
                of the rotor diameter.
            distance_pmf (dict, optional): Probability mass function describing the
                length of steps in the random search. Specified as a dictionary with 
                keys "d" (array of step distances, specified in meters) and "p" 
                (array of probability of occurence, should sum to 1). Defaults to 
                uniform probability between 0.5D and 2D, with some extra mass
                to encourage large changes. 
            n_individuals (int, optional): The number of individuals to use in the
                optimization. Defaults to 4.
            seconds_per_iteration (float, optional): The number of seconds to
                run each step of the optimization for. Defaults to 60.
            total_optimization_seconds (float, optional): The total number of
                seconds to run the optimization for. Defaults to 600.
            interface (str): Parallel computing interface to leverage. Recommended is 'concurrent'
                or 'multiprocessing' for local (single-system) use, and 'mpi4py' for high
                performance computing on multiple nodes. Defaults to 'multiprocessing'.
            max_workers (int): Number of parallel workers, typically equal to the number of cores
                you have on your system or HPC.  Defaults to None, which will use all
                available cores.
            grid_step_size (float): The courseness of the grid used to generate the initial layout.
                Defaults to 100.
            relegation_number (int): The number of the lowest performing individuals to be replaced
                with new individuals generated from the best performing individual.  Must
                be less than n_individuals / 2.  Defaults to 1.
            enable_geometric_yaw (bool): Use geometric yaw code to determine approximate wake 
                steering yaw angles during layout optimization routine. Defaults to False.
            use_dist_based_init (bool): Generate initial layouts automatically by placing turbines
                as far apart as possible.
            random_seed (int or None): Random seed for reproducibility. Defaults to None.
        """
        # The parallel computing interface to use
        if interface == "mpi4py":
            import mpi4py.futures as mp
            self._PoolExecutor = mp.MPIPoolExecutor
        elif interface == "multiprocessing":
            import multiprocessing as mp
            self._PoolExecutor = mp.Pool
            if max_workers is None:
                max_workers = mp.cpu_count()
        elif interface is None:
            if n_individuals > 1 or (max_workers is not None and max_workers > 1):
                print("Parallelization not possible with interface=None. "+\
                    "Reducing n_individuals to 1 and ignoring max_workers.")
                self._PoolExecutor = None
                max_workers = None
                n_individuals = 1

            
        # elif interface == "concurrent":
        #     from concurrent.futures import ProcessPoolExecutor
        #     self._PoolExecutor = ProcessPoolExecutor
        else:
            raise UserWarning(
                f"Interface '{interface}' not recognized. "
                "Please use ' 'multiprocessing' or 'mpi4py'."
            )

        # Store the max_workers
        self.max_workers = max_workers

        # Store the interface
        self.interface = interface

        # Set and store the random seed
        self.random_seed = random_seed

        # Confirm the relegation_number is valid
        if relegation_number > n_individuals / 2:
            raise ValueError("relegation_number must be less than n_individuals / 2.")
        self.relegation_number = relegation_number

        # Store the rotor diameter and number of turbines
        self.D = fi.floris.farm.rotor_diameters_sorted[0][0][0]
        self.N_turbines = fi.floris.farm.n_turbines

        # Make sure not both min_dist and min_dist_D are defined
        if min_dist is not None and min_dist_D is not None:
            raise ValueError("Only one of min_dist and min_dist_D can be defined.")

        # If min_dist_D is defined, convert to min_dist
        if min_dist_D is not None:
            min_dist = min_dist_D * self.D

        super().__init__(fi, boundaries, min_dist=min_dist, freq=freq,
            enable_geometric_yaw=enable_geometric_yaw)

        # Save min_dist_D
        self.min_dist_D = self.min_dist / self.D

        # Process and save the step distribution
        self._process_dist_pmf(distance_pmf)

        # Store the fi_dict
        self.fi_dict = self.fi.floris.as_dict()

        # Save the grid step size
        self.grid_step_size = grid_step_size

        # Save number of individuals
        self.n_individuals = n_individuals

        # Store the initial locations
        self.x_initial = self.fi.layout_x
        self.y_initial = self.fi.layout_y

        # Store the total optimization seconds
        self.total_optimization_seconds = total_optimization_seconds

        # Store the seconds per iteration
        self.seconds_per_iteration = seconds_per_iteration

        # Get the initial AEP value
        self.x = self.x_initial # Required by _get_geoyaw_angles
        self.y = self.y_initial # Required by _get_geoyaw_angles
        self.aep_initial = _get_aep(
            self.x_initial,
            self.y_initial,
            self.fi,
            self.freq,
            self._get_geoyaw_angles()
        )
        #del self.x # Deleting to avoid confusion, since not updated
        #del self.y # Deleting to avoid confusion, since not updated

        # Initialize the aep statistics
        self.aep_mean = self.aep_initial
        self.aep_median = self.aep_initial
        self.aep_max = self.aep_initial
        self.aep_min = self.aep_initial

        # Initialize the numpy arrays which will hold the candidate layouts
        # these will have dimensions n_individuals x N_turbines
        self.x_candidate = np.zeros((self.n_individuals, self.N_turbines))
        self.y_candidate = np.zeros((self.n_individuals, self.N_turbines))

        # Initialize the array which will hold the AEP values for each candidate
        self.aep_candidate = np.zeros(self.n_individuals)

        # Initialize the iteration step
        self.iteration_step = -1

        # Initialize the optimization time
        self.opt_time_start = timerpc()
        self.opt_time = 0

        # Generate the initial layouts
        if use_dist_based_init:
            self._generate_initial_layouts()
        else:
            print(f'Using supplied initial layout for {self.n_individuals} individuals.')
            for i in range(self.n_individuals):
                self.x_candidate[i, :] = self.x_initial
                self.y_candidate[i, :] = self.y_initial
                self.aep_candidate[i] = self.aep_initial

        # Evaluate the initial optimization step
        self._evaluate_opt_step()

    def describe(self):
        print("Random Layout Optimization")
        print(f"Number of turbines to optimize = {self.N_turbines}")
        print(f"Minimum distance between turbines = {self.min_dist_D} [D], {self.min_dist} [m]")
        print(f"Number of individuals = {self.n_individuals}")
        print(f"Seconds per iteration = {self.seconds_per_iteration}")
        print(f"Initial AEP = {self.aep_initial/1e9:.1f} [GWh]")

    def _process_dist_pmf(self, dist_pmf):
        """
        Check validity of pmf and assign default if none provided.
        """
        if dist_pmf is None:
            jump_dist = np.min([self.xmax-self.xmin, self.ymax-self.ymin])/2
            jump_prob = 0.05

            d = np.append(np.linspace(0.5*self.D, 2.0*self.D, 151), jump_dist)
            p = np.append((1-jump_prob)/len(d)*np.ones(len(d)-1), jump_prob)
            p = p / p.sum()
            dist_pmf = {"d":d, "p":p}

        # Check correct keys are provided
        if not all(k in dist_pmf for k in ("d", "p")):
            raise KeyError("distance_pmf must contains keys \"d\" (step distance)"+\
                " and \"p\" (probability of occurance).")
        
        # Check entries are in the correct form
        if not hasattr(dist_pmf["d"], "__len__") or not hasattr(dist_pmf["d"], "__len__")\
            or len(dist_pmf["d"]) != len(dist_pmf["p"]):
            raise TypeError("distance_pmf entries should be numpy arrays or lists"+\
                " of equal length.")

        if np.sum(dist_pmf["p"]) != 1:
            print("Probability mass function does not sum to 1. Normalizing.")
            dist_pmf["p"] = np.array(dist_pmf["p"]) / np.array(dist_pmf["p"]).sum()

        self.distance_pmf = dist_pmf

    def _evaluate_opt_step(self):

        # Sort the candidate layouts by AEP
        sorted_indices = np.argsort(self.aep_candidate)[::-1] # Decreasing order
        self.aep_candidate = self.aep_candidate[sorted_indices]
        self.x_candidate = self.x_candidate[sorted_indices]
        self.y_candidate = self.y_candidate[sorted_indices]

        # Update the optimization time
        self.opt_time = timerpc() - self.opt_time_start

        # Update the optimizations step
        self.iteration_step += 1

        # Update the AEP statistics
        self.aep_mean = np.mean(self.aep_candidate)
        self.aep_median = np.median(self.aep_candidate)
        self.aep_max = np.max(self.aep_candidate)
        self.aep_min = np.min(self.aep_candidate)

        # Report the results
        print("=======================================")
        print(f"Optimization step {self.iteration_step:+.1f}")
        print(f"Optimization time = {self.opt_time:+.1f} [s]")
        print(f"Mean AEP = {self.aep_mean/1e9:.1f} [GWh] \
              ({100 * (self.aep_mean - self.aep_initial) / self.aep_initial:+.2f}%)")
        print(f"Median AEP = {self.aep_median/1e9:.1f} [GWh] \
               ({100 * (self.aep_median - self.aep_initial) / self.aep_initial:+.2f}%)")
        print(f"Max AEP = {self.aep_max/1e9:.1f} [GWh] \
              ({100 * (self.aep_max - self.aep_initial) / self.aep_initial:+.2f}%)")
        print(f"Min AEP = {self.aep_min/1e9:.1f} [GWh] \
               ({100 * (self.aep_min - self.aep_initial) / self.aep_initial:+.2f}%)")
        print("=======================================")

        # Replace the relegation_number worst performing layouts with relegation_number
        # best layouts
        if self.relegation_number > 0:
            self.aep_candidate[-self.relegation_number:] = self.aep_candidate[:self.relegation_number]
            self.x_candidate[-self.relegation_number:] = self.x_candidate[:self.relegation_number]
            self.y_candidate[-self.relegation_number:] = self.y_candidate[:self.relegation_number]


    # Private methods
    def _generate_initial_layouts(self):
        """
        This method generates n_individuals initial layout of turbines. It does
        this by calling the _generate_random_layout method within a multiprocessing
        pool.
        """

        # Set random seed for initial layout
        if self.random_seed is None:
            multi_random_seeds = [None]*self.n_individuals
        else:
            multi_random_seeds = [23 + i for i in range(self.n_individuals)]
            # 23 is just an arbitrary choice to ensure different random seeds
            # to the evaluation code
            
        print(f'Generating {self.n_individuals} initial layouts...')
        t1 = timerpc()
        # Generate the multiargs for parallel execution
        multiargs = [
            (self.N_turbines,
            self.grid_step_size,
            self._boundary_polygon,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            multi_random_seeds[i])
            for i in range(self.n_individuals)
        ]

        if self._PoolExecutor: # Parallelized
            with self._PoolExecutor(self.max_workers) as p:
                # This code is not currently necessary, but leaving in case implement
                # concurrent later, based on parallel_computing_interface.py
                if (self.interface == "mpi4py") or (self.interface == "multiprocessing"):
                        out = p.starmap(_gen_dist_based_init, multiargs)
        else: # Parallelization not activated
            out = [_gen_dist_based_init(*multiargs[0])]

        # Unpack out into the candidate layouts
        for i in range(self.n_individuals):
            self.x_candidate[i, :] = out[i][0]
            self.y_candidate[i, :] = out[i][1]

        # Get the AEP values for each candidate layout
        for i in range(self.n_individuals):
            self.aep_candidate[i] = _get_aep(
                self.x_candidate[i, :],
                self.y_candidate[i, :],
                self.fi,
                self.freq,
                self._get_geoyaw_angles()
            )

        t2 = timerpc()
        print(f"  Time to generate intial layouts: {t2-t1:.3f} s")
        print(len(out))
        print(out)



    def _get_initial_and_final_locs(self):
        x_initial = self.x_initial
        y_initial = self.y_initial
        x_opt = self.x_opt
        y_opt = self.y_opt
        return x_initial, y_initial, x_opt, y_opt


    # Public methods

    def optimize(self):
        """
        Perform the optimization
        """
        print(f'Optimizing using {self.n_individuals} individuals.')
        opt_start_time = timerpc()
        opt_stop_time = opt_start_time + self.total_optimization_seconds
        sim_time = 0

        self.aep_candidate_log = [self.aep_candidate] # TODO: Not recording initial correctly?
        self.num_aep_calls_log = []
        self._num_aep_calls = [0]*self.n_individuals

        while timerpc() < opt_stop_time:

            # Set random seed for the main loop
            if self.random_seed is None:
                multi_random_seeds = [None]*self.n_individuals
            else:
                multi_random_seeds = [55 + self.iteration_step + i 
                    for i in range(self.n_individuals)]
            # 55 is just an arbitrary choice to ensure different random seeds
            # to the initialization code

            # Update the optimization time
            sim_time = timerpc() - opt_start_time
            print(f'Optimization time: {sim_time:.1f} s / {self.total_optimization_seconds:.1f} s')


            # Generate the multiargs for parallel execution of single individual optimization
            multiargs = [
                (self.seconds_per_iteration,
                 self.aep_candidate[i],
                 self.x_candidate[i, :],
                 self.y_candidate[i, :],
                 self.fi_dict,
                 self.freq,
                 self.min_dist,
                 self._boundary_polygon,
                 self.distance_pmf,
                 self.enable_geometric_yaw,
                 multi_random_seeds[i]
                )
                    for i in range(self.n_individuals)
            ]

            # Run the single individual optimization in parallel
            if self._PoolExecutor: # Parallelized
                with self._PoolExecutor(self.max_workers) as p:
                    out = p.starmap(_single_individual_opt, multiargs)
            else: # Parallelization not activated
                out = [_single_individual_opt(*multiargs[0])]

            # Unpack the results
            for i in range(self.n_individuals):
                self.aep_candidate[i] = out[i][0]
                self.x_candidate[i, :] = out[i][1]
                self.y_candidate[i, :] = out[i][2]
                self._num_aep_calls[i] = out[i][3]
            self.aep_candidate_log.append(self.aep_candidate)
            self.num_aep_calls_log.append(self._num_aep_calls)

            # Evaluate the individuals for this step
            self._evaluate_opt_step()

        # Finalize the result
        self.final_aep = self.aep_candidate[0]
        self.x_opt = self.x_candidate[0, :]
        self.y_opt = self.y_candidate[0, :]

        # Print the final result
        print(f'Final AEP = {self.final_aep/1e9:.1f} [GWh] \
               ({100 * (self.final_aep - self.aep_initial) / self.aep_initial:+.2f}%)')

        return self.final_aep, self.x_opt, self.y_opt

    
    # Helpful visualizations
    def plot_distance_pmf(self, ax=None):
        """
        Tool to check the used distance pmf.
        """

        if ax is None:
            fig, ax = plt.subplots(1,1)
        
        ax.stem(self.distance_pmf["d"], self.distance_pmf["p"], linefmt="k-")
        ax.grid(True)
        ax.set_xlabel("Step distance [m]")
        ax.set_ylabel("Probability")

        return ax



def _single_individual_opt(
    seconds_per_iteration,
    initial_aep,
    layout_x,
    layout_y,
    fi_dict,
    freq,
    min_dist,
    poly_outer,
    dist_pmf,
    enable_geometric_yaw,
    s
):
    # Set random seed
    np.random.seed(s)

    # Initialize the optimization time
    single_opt_start_time = timerpc()
    stop_time = single_opt_start_time + seconds_per_iteration

    num_aep_calls = 0

    # Get the fi
    fi_ = _load_local_floris_object(fi_dict)

    # Initialize local variables
    num_turbines = len(layout_x)
    get_new_point = True # TODO: CHECK: how useful is this?
    current_aep = initial_aep

    # Establish geometric yaw optimizer, if desired
    if enable_geometric_yaw:
        yaw_opt = YawOptimizationGeometric(
            fi_,
            minimum_yaw_angle=-30.0,
            maximum_yaw_angle=30.0,
            exploit_layout_symmetry=False
        )
    else: # yaw_angles will always be none
        yaw_angles = None

    # Loop as long as we've not hit the stop time
    while timerpc() < stop_time:

            if get_new_point: #If the last test wasn't succesfull

                # Randomly select a turbine to nudge
                tr = random.randint(0,num_turbines-1)

                # Randomly select a direction to nudge in (uniform direction)
                rand_dir = np.random.uniform(low=0.0, high=2*np.pi)

                # Randomly select a distance to travel according to pmf
                rand_dist = np.random.choice(dist_pmf["d"], p=dist_pmf["p"])

            # Get a new test point
            test_x = layout_x[tr] + np.cos(rand_dir) * rand_dist
            test_y = layout_y[tr] + np.sin(rand_dir) * rand_dist

            # In bounds?
            if not test_point_in_bounds(test_x, test_y, poly_outer):
                get_new_point = True
                continue

            # Make a new layout
            original_x = layout_x[tr]
            original_y = layout_y[tr]
            layout_x[tr] = test_x
            layout_y[tr] = test_y

            # Acceptable distances?
            if not test_min_dist(layout_x, layout_y,min_dist):
                # Revert and continue
                layout_x[tr] = original_x
                layout_y[tr] = original_y
                get_new_point = True
                continue

            # Does it improve AEP?
            if enable_geometric_yaw: # Select appropriate yaw angles
                yaw_opt.fi_subset.reinitialize(layout_x=layout_x, layout_y=layout_y)
                df_opt = yaw_opt.optimize()
                yaw_angles = np.vstack(df_opt['yaw_angles_opt'])[:, None, :]

            num_aep_calls += 1
            test_aep = _get_aep(layout_x, layout_y, fi_, freq, yaw_angles) 
            # TODO: Geoyaw angles not available here!

            if test_aep > current_aep:
                # Accept the change
                current_aep = test_aep

                # If not a random point this cycle and it did improve things
                # try not getting a new point
                get_new_point = False # TODO: Possibly remove this feature

            else:
                # Revert the change
                layout_x[tr] = original_x
                layout_y[tr] = original_y
                get_new_point = True # TODO: Possibly remove this feature
                continue

    # Return the best result from this individual
    return current_aep, layout_x, layout_y, num_aep_calls
