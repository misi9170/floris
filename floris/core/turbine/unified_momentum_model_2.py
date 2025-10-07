
from __future__ import annotations

import sys

import numpy as np

from floris.core.rotor_velocity import (
    average_velocity,
    rotor_velocity_air_density_correction,
)
from floris.core.turbine.operation_models import BaseOperationModel
from floris.type_dec import NDArrayFloat

# Cache the import result at module level
_mr = None
_mr_available = None

def _import_MITRotor_package():
    global _mr, _mr_available
    if _mr_available is None:
        try:
            import MITRotor as mr
            _mr = mr
            _mr_available = True
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "MITRotor is required for this module. Install it with:",
                " pip install git+https://github.com/Howland-Lab/MITRotor.git"
            )
    return _mr

def _construct_bem():
    rotor = _mr.IEA10MW() # TODO: Hardcoded
    geometry = _mr.BEMGeometry(Nr=20, Ntheta=10) # TODO: Hardcoded
    bem = _mr.BEM(rotor=rotor, geometry=geometry)

    return bem

class UnifiedMomentumModelTurbine_2(BaseOperationModel):
    """
    Turbine operation model as described by Heck et al. (2023). [Check]
    """

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **kwargs,
    ) -> None:

        _import_MITRotor_package()

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )
        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        # Construct the MITRotor BEM solver
        bem = _construct_bem()

        # Determine operating point (Pitch, TSR)
        # TODO: Handle findices (look at UMMTurbine?)
        sol = bem(pitch=0.0, tsr=7.0, yaw=yaw_angles)  # TODO: Hardcoded

        if not sol.converged:
            # TODO: error? Or warning?
            raise RuntimeError("BEM solution did not converge.")

        # Compute power
        power_coefficient = sol.Cp(grid="rotor")
        power = 0.5 * air_density * np.pi * bem.rotor.R**2 * power_coefficient * rotor_effective_velocities**3

        return power

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **kwargs,
    ) -> None:
        _import_MITRotor_package()

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )
        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        # Construct the MITRotor BEM solver
        bem = _construct_bem()

        # Determine operating point (Pitch, TSR)
        # TODO: Handle findices (look at UMMTurbine?)
        sol = bem(pitch=0.0, tsr=7.0, yaw=yaw_angles)  # TODO: Hardcoded

        if not sol.converged:
            # TODO: error? Or warning?
            raise RuntimeError("BEM solution did not converge.")

        # Return thrust coefficient
        return sol.Ct(grid="rotor")

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **kwargs,
    ):
        _import_MITRotor_package()

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )
        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        # Construct the MITRotor BEM solver
        bem = _construct_bem()

        # Determine operating point (Pitch, TSR)
        # TODO: Handle findices (look at UMMTurbine?)
        # TODO: Check if op point is different, save in memory to speed up?
        sol = bem(pitch=0.0, tsr=7.0, yaw=yaw_angles)  # TODO: Hardcoded

        if not sol.converged:
            # TODO: error? Or warning?
            raise RuntimeError("BEM solution did not converge.")

        # Return solution
        return sol.a(grid="rotor")
