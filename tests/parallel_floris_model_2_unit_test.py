
import copy
import logging

import numpy as np
import pytest

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.parallel_floris_model_2 import ParallelFlorisModel


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

def test_None_interface(sample_inputs_fixture):
    """
    With interface=None, the ParallelFlorisModel should behave exactly like the FlorisModel.
    (ParallelFlorisModel.run() simply calls the parent FlorisModel.run()).
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface=None,
        n_wind_condition_splits=2 # Not used when interface=None
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_multiprocessing_interface(sample_inputs_fixture):
    """
    With interface="multiprocessing", the ParallelFlorisModel should return the same powers
    as the FlorisModel.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_return_turbine_powers_only(sample_inputs_fixture):
    """
    With return_turbine_powers_only=True, the ParallelFlorisModel should return only the
    turbine powers, not the full results.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2,
        return_turbine_powers_only=True
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_run_error(sample_inputs_fixture, caplog):
    """
    Check that an error is raised if an output is requested before calling run().
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2
    )

    # In future versions, error will be raised
    # with pytest.raises(RuntimeError):
    #     pfmodel.get_turbine_powers()
    # with pytest.raises(RuntimeError):
    #     pfmodel.get_farm_AEP()

    # For now, only a warning is raised for backwards compatibility
    with caplog.at_level(logging.WARNING):
        pfmodel.get_turbine_powers()
    assert caplog.text != "" # Checking not empty
    caplog.clear()

def test_configuration_compatibility(sample_inputs_fixture, caplog):
    """
    Check that the ParallelFlorisModel is compatible with FlorisModel and
    UncertainFlorisModel configurations.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)

    with caplog.at_level(logging.WARNING):
        ParallelFlorisModel(fmodel)
    assert caplog.text != "" # Checking not empty
    caplog.clear()

    pfmodel = ParallelFlorisModel(sample_inputs_fixture.core)
    with caplog.at_level(logging.WARNING):
        pfmodel.fmodel
    assert caplog.text != "" # Checking not empty
    caplog.clear()

    with pytest.raises(AttributeError):
        pfmodel.fmodel.core

def test_wind_data_objects(sample_inputs_fixture):
    """
    Check that the ParallelFlorisModel is compatible with WindData objects.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParallelFlorisModel(sample_inputs_fixture.core)

    # Create a wind rose and set onto both models
    wind_speeds = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 280.0, 280.0, 280.0])
    wind_rose = WindRose(
        wind_directions=np.unique(wind_directions),
        wind_speeds=np.unique(wind_speeds),
        ti_table=0.06
    )
    fmodel.set(wind_data=wind_rose)
    pfmodel.set(wind_data=wind_rose)

    # Run; get turbine powers; compare results
    fmodel.run()
    powers_fmodel_wr = fmodel.get_turbine_powers()
    pfmodel.run()
    powers_pfmodel_wr = pfmodel.get_turbine_powers()

    assert powers_fmodel_wr.shape == powers_pfmodel_wr.shape
    assert np.allclose(powers_fmodel_wr, powers_pfmodel_wr)

    # Test a TimeSeries object
    wind_speeds = np.array([8.0, 8.0, 9.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    values = np.array([30.0, 20.0, 10.0])
    time_series = TimeSeries(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=0.06,
        values=values,
    )
    fmodel.set(wind_data=time_series)
    pfmodel.set(wind_data=time_series)

    fmodel.run()
    powers_fmodel_ts = fmodel.get_turbine_powers()
    pfmodel.run()
    powers_pfmodel_ts = pfmodel.get_turbine_powers()

    assert powers_fmodel_ts.shape == powers_pfmodel_ts.shape
    assert np.allclose(powers_fmodel_ts, powers_pfmodel_ts)
