import logging
from pathlib import Path

import numpy as np
import pytest

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.optimization.layout_optimization.layout_optimization_base import (
    LayoutOptimization,
)
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)
from floris.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
from floris.optimization.layout_optimization.layout_optimization_gridded import (
    LayoutOptimizationGridded,
)
from floris.wind_data import WindDataBase


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

test_boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]


def test_base_class(caplog):
    # Get a test fi
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Now initiate layout optimization with a frequency matrix passed in the 3rd position
    # (this should fail)
    freq = np.ones((5, 5))
    freq = freq / freq.sum()

    # Check that warning is raised if fmodel does not contain wind_data
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel, test_boundaries, 5)
    assert caplog.text != "" # Checking not empty

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel=fmodel, boundaries=test_boundaries, min_dist=5,)
    assert caplog.text != "" # Checking not empty

    time_series = TimeSeries(
        wind_directions=fmodel.core.flow_field.wind_directions,
        wind_speeds=fmodel.core.flow_field.wind_speeds,
        turbulence_intensities=fmodel.core.flow_field.turbulence_intensities,
    )
    fmodel.set(wind_data=time_series)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel, test_boundaries, 5)
    assert caplog.text != "" # Not empty, because get_farm_AEP called on TimeSeries

    # Passing without keyword arguments should work, or with keyword arguments
    LayoutOptimization(fmodel, test_boundaries, 5)
    LayoutOptimization(fmodel=fmodel, boundaries=test_boundaries, min_dist=5)

    # Check with WindRose on fmodel
    fmodel.set(wind_data=time_series.to_WindRose())

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel, test_boundaries, 5)
    assert caplog.text == "" # Empty

    LayoutOptimization(fmodel, test_boundaries, 5)
    LayoutOptimization(fmodel=fmodel, boundaries=test_boundaries, min_dist=5)

def test_LayoutOptimizationRandomSearch():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(layout_x=[0, 500], layout_y=[0, 0])

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel=fmodel,
        boundaries=test_boundaries,
        min_dist_D=5,
        seconds_per_iteration=1,
        total_optimization_seconds=1,
        use_dist_based_init=False,
    )

    # Check that the optimization runs
    layout_opt.optimize()

def test_LayoutOptimizationGridded_initialization(caplog):
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(layout_x=[0, 500], layout_y=[0, 0])

    with pytest.raises(ValueError):
        LayoutOptimizationGridded(
            fmodel=fmodel,
            boundaries=test_boundaries,
            spacing=None,
            spacing_D=None,
        ) # No spacing specified
    with pytest.raises(ValueError):
        LayoutOptimizationGridded(
            fmodel=fmodel,
            boundaries=test_boundaries,
            spacing=500,
            spacing_D=5
        ) # Spacing specified in two ways
    
    fmodel.core.farm.rotor_diameters[1] = 100.0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        LayoutOptimizationGridded(
            fmodel,
            test_boundaries,
            spacing_D=5
        )

def test_LayoutOptimizationGridded_default_grid():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set up a sample boundary
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    layout_opt = LayoutOptimizationGridded(
        fmodel=fmodel,
        boundaries=boundaries,
        spacing=50,
    )

    # Test it worked...

def test_LayoutOptimizationGridded_basic():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    spacing = 60

    layout_opt = LayoutOptimizationGridded(
        fmodel=fmodel,
        boundaries=test_boundaries,
        spacing=spacing,
        rotation_step=5,
        rotation_range=(0, 360),
        translation_step=50,
        hexagonal_packing=False,
        enable_geometric_yaw=False,
        use_value=False,
    )

    n_turbs_opt, x_opt, y_opt = layout_opt.optimize()

    # Check that the number of turbines is correct
    assert n_turbs_opt == len(x_opt)


    # Check all are indeed in bounds
    assert (np.all(x_opt > 0.0) & np.all(x_opt < 1000.0)
            & np.all(y_opt > 0.0) & np.all(y_opt < 1000.0))

    # Check that the layout is at least as good as the basic rectangular fill
    n_turbs_subopt = (1000 // spacing + 1) ** 2

    assert n_turbs_opt >= n_turbs_subopt

def test_LayoutOptimizationGridded_diagonal():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    turbine_spacing = 1000.0
    corner = 2*turbine_spacing / np.sqrt(2)

    # Create a "thin" boundary area at a 45 degree angle
    boundaries_diag = [
        (0.0, 0.0),
        (0.0, 10.0),
        (corner, corner+10),
        (corner+10, corner+10),
        (0.0, 0.0)
    ]

    layout_opt = LayoutOptimizationGridded(
        fmodel=fmodel,
        boundaries=boundaries_diag,
        spacing=turbine_spacing,
        rotation_step=5,
        rotation_range=(0, 360),
        translation_step=1,
        hexagonal_packing=False,
        enable_geometric_yaw=False,
        use_value=False,
    )

    n_turbs_opt, _, _ = layout_opt.optimize()
    assert n_turbs_opt == 3 # 3 should fit in the diagonal

    # Also test a limited rotation; should be worse.
    # Also test a very coarse rotation step; should be worse.
    # Also test a very coarse translation step; should be worse.
