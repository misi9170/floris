from pathlib import Path

import numpy as np

from floris import FlorisModel
from floris.turbine_library import build_cosine_loss_turbine_dict


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

def test_row_of_turbines():

    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Configure as turboparkgauss
    fmodel_dict = fmodel.core.as_dict()
    fmodel_dict["wake"]["model_strings"]["velocity_model"] = "turboparkgauss"
    fmodel_dict["wake"]["model_strings"]["turbulence_model"] = "none"
    fmodel_dict["wake"]["model_strings"]["deflection_model"] = "none"
    fmodel_dict["wake"]["model_strings"]["combination_model"] = "sosfs"
    fmodel_dict["wake"]["enable_secondary_steering"] = False
    fmodel_dict["wake"]["enable_yaw_added_recovery"] = False
    fmodel_dict["wake"]["enable_active_wake_mixing"] = False
    fmodel_dict["wake"]["enable_transverse_velocities"] = False
    fmodel_dict["solver"]["type"] = "turbine_cubature_grid"
    fmodel_dict["solver"]["turbine_grid_points"] = 6
    fmodel = FlorisModel(configuration=fmodel_dict)

    # Define turbine
    const_CT_turb = build_cosine_loss_turbine_dict(
        turbine_data_dict={
            "wind_speed":[0.0, 30.0],
            "power":[0.0, 1.0], # Not realistic but won't be used here
            "thrust_coefficient":[0.75, 0.75]
        },
        turbine_name="ConstantCT",
        rotor_diameter=120.0,
        hub_height=100.0,
        ref_tilt=0.0,
    )

    # Set up problem and run
    fmodel.set(
        layout_x=np.linspace(0.0, 5400.0, 10),
        layout_y=np.zeros(10),
        wind_speeds=[8.0],
        wind_directions=[270.0],
        turbulence_intensities=[0.06],
        wind_shear=0.0,
        turbine_type=[const_CT_turb],
    )
    fmodel.run()

    # Run and extract flow velocities at the turbines
    velocities_row_normalized = fmodel.turbine_average_velocities[0,:] / 8.0

    # Comparison data from Nygaard / Orsted
    velocities_comparison = np.array([
        1.0,
        0.709920677983239,
        0.615355749367675,
        0.551410465937128,
        0.502600655337247,
        0.463167556093190,
        0.430238792036599,
        0.402137593655074,
        0.377783142608699,
        0.356429516711137,
    ])

    # Compare the results
    print(velocities_row_normalized)

    np.testing.assert_allclose(
        velocities_row_normalized,
        velocities_comparison,
        rtol=1e-2,
    ) # Within 1% tolerance