"""Example: Compare parallel interfaces
"""

from time import perf_counter as timerpc

import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    UncertainFlorisModel,
    WindRose,
)
# from floris.parallel_floris_model import ParallelFlorisModel as ParallelFlorisModel_orig
from floris.parallel_floris_model_2 import ParallelFlorisModel as ParallelFlorisModel_new


if __name__ == "__main__":
    # Parallelization parameters
    parallel_interface = "multiprocessing"
    max_workers = 16

    # Load the wind rose from csv
    wind_rose = WindRose.read_csv_long(
        "inputs/wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val",
        ti_col_or_value=0.06
    )
    fmodel = FlorisModel("inputs/gch.yaml")

    # Specify wind farm layout and update in the floris object
    N = 3  # number of turbines per row and per column
    X, Y = np.meshgrid(
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
    )
    fmodel.set(layout_x=X.flatten(), layout_y=Y.flatten())


    # Set up new parallel Floris model
    pfmodel_new = ParallelFlorisModel_new(
        "inputs/gch.yaml",
        max_workers=max_workers,
        n_wind_condition_splits=max_workers,
        interface=parallel_interface,
        print_timings=True,
    )

    # Set layout, wind data on all models
    fmodel.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=wind_rose)
    pfmodel_new.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=wind_rose)

    # Set up the uncertain model using the fmodel and pfmodel_new
    ufmodel = UncertainFlorisModel(fmodel, wd_std=3.0)
    upfmodel = UncertainFlorisModel(pfmodel_new, wd_std=3.0)



    # Run and evaluate farm over the wind rose
    ufmodel.run()
    aep_fmodel = ufmodel.get_farm_AEP()

    upfmodel.run()
    aep_upfmodel = upfmodel.get_farm_AEP()


    print(f"AEP for fmodel: {aep_fmodel}")
    print(f"AEP for pfmodel_new: {aep_upfmodel}")
