# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import define, field

from floris.simulation import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import (
    cosd,
    sind,
    tand,
)


@define
class GaussVelocityDeficit(BaseModel):

    alpha: float = field(default=0.58)
    beta: float = field(default=0.077)
    ka: float = field(default=0.38)
    kb: float = field(default=0.004)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "u_initial": flow_field.u_initial_sorted,
            "wind_veer": flow_field.wind_veer
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: float
    ) -> None:

        # yaw_angle is all turbine yaw angles for each wind speed
        # Extract and broadcast only the current turbine yaw setting
        # for all wind speeds

        # Opposite sign convention in this model
        yaw_angle = -1 * yaw_angle_i

        # Initialize the velocity deficit
        uR = u_initial * ct_i / ( 2.0 * (1 - np.sqrt(1 - ct_i) ) )
        u0 = u_initial * np.sqrt(1 - ct_i)

        # Initial lateral bounds
        sigma_z0 = rotor_diameter_i * 0.5 * np.sqrt(uR / (u_initial + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_angle) * cosd(wind_veer)


        # Compute the bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = x_i

        # Start of the far wake
        x0 = np.ones_like(u_initial)
        x0 *= rotor_diameter_i * cosd(yaw_angle) * (1 + np.sqrt(1 - ct_i) )
        x0 /= np.sqrt(2) * (
            4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i) )
        )
        x0 += x_i

        # Initialize the velocity deficit array
        velocity_deficit = np.zeros_like(u_initial)

        # Masks
        # When we have only an inequality, the current turbine may be applied its own
        # wake in cases where numerical precision cause in incorrect comparison. We've
        # applied a small bump to avoid this. "0.1" is arbitrary but it is a small, non
        # zero value.

        # This mask defines the near wake; keeps the areas downstream of xR and upstream of x0
        near_wake_mask = np.array(x > xR + 0.1) * np.array(x < x0)
        far_wake_mask = np.array(x >= x0)

        # Compute the velocity deficit in the NEAR WAKE region
        # ONLY If there are points within the near wake boundary
        # TODO: for the turbinegrid, do we need to do this near wake calculation at all?
        #       same question for any grid with a resolution larger than the near wake region
        if np.sum(near_wake_mask):

            # Calculate the wake expansion

            # This is a linear ramp from 0 to 1 from the start of the near wake to the start
            # of the far wake.
            near_wake_ramp_up = (x - xR) / (x0 - xR)
            # Another linear ramp, but positive upstream of the far wake and negative in the
            # far wake; 0 at the start of the far wake
            near_wake_ramp_down = (x0 - x) / (x0 - xR)
            # near_wake_ramp_down = -1 * (near_wake_ramp_up - 1)  # TODO: this is equivalent, right?

            sigma_y = near_wake_ramp_down * 0.501 * rotor_diameter_i * np.sqrt(ct_i / 2.0)
            sigma_y += near_wake_ramp_up * sigma_y0
            sigma_y *= np.array(x >= xR)
            sigma_y += np.ones_like(sigma_y) * np.array(x < xR) * 0.5 * rotor_diameter_i

            sigma_z = near_wake_ramp_down * 0.501 * rotor_diameter_i * np.sqrt(ct_i / 2.0)
            sigma_z += near_wake_ramp_up * sigma_z0
            sigma_z *= np.array(x >= xR)
            sigma_z += np.ones_like(sigma_z) * np.array(x < xR) * 0.5 * rotor_diameter_i

            r, C = rC(
                wind_veer,
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_i,
                z,
                hub_height_i,
                ct_i,
                yaw_angle,
                rotor_diameter_i
            )

            near_wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
            near_wake_deficit *= near_wake_mask

            velocity_deficit += near_wake_deficit


        # Compute the velocity deficit in the FAR WAKE region
        if np.sum(far_wake_mask):

            # Wake expansion in the lateral (y) and the vertical (z)
            ky = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            kz = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            sigma_y = (ky * (x - x0) + sigma_y0) * far_wake_mask + sigma_y0 * np.array(x < x0)
            sigma_z = (kz * (x - x0) + sigma_z0) * far_wake_mask + sigma_z0 * np.array(x < x0)

            r, C = rC(
                wind_veer,
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_i,
                z,
                hub_height_i,
                ct_i,
                yaw_angle,
                rotor_diameter_i
            )

            far_wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
            far_wake_deficit *= far_wake_mask

            velocity_deficit += far_wake_deficit

        return velocity_deficit

@define
class GaussGeometricVelocityDeficit(BaseModel):

    wake_expansion_rates: list = field(default=[0.0, 0.012])
    breakpoints_D: list = field(default=[5])
    sigma_y0_D: float = field(default=0.28)
    smoothing_length_D: float = field(default=2.0)
    mixing_gain_velocity: float = field(default=2.5) 

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = dict(
            x=grid.x_sorted,
            y=grid.y_sorted,
            z=grid.z_sorted,
            u_initial=flow_field.u_initial_sorted,
            wind_veer=flow_field.wind_veer
        )
        return kwargs

    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_y_i: np.ndarray,
        deflection_field_z_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        tilt_angle_i: np.ndarray,
        mixing_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: float
    ) -> None:

        include_mirror_wake = True # Could add this as a user preference.

        # yaw_angle is all turbine yaw angles for each wind speed
        # Extract and broadcast only the current turbine yaw setting
        # for all wind speeds

        # Only symmetric terms using yaw in this model, but anyway:
        yaw_angle = -1 * yaw_angle_i

        # Initialize the velocity deficit
        uR = u_initial * ct_i / ( 2.0 * (1 - np.sqrt(1 - ct_i) ) )
        u0 = u_initial * np.sqrt(1 - ct_i)

        # Initial lateral bounds
        #sigma_z0 = rotor_diameter_i * 0.5 * np.sqrt(uR / (u_initial + u0))
        # TODO: add a yawing component?
        sigma_y0 = self.sigma_y0_D * rotor_diameter_i * cosd(yaw_angle)
        sigma_z0 = self.sigma_y0_D * rotor_diameter_i * cosd(tilt_angle_i)

        # No specific near, far wakes in this model
        downstream_mask = np.array(x > x_i + 0.1)
        upstream_mask = np.array(x < x_i - 0.1)

        # Wake expansion in the lateral (y) and the vertical (z)
        sigma_y = geometric_model_wake_width(
            x-x_i, 
            self.wake_expansion_rates, 
            [b*rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_y0, 
            self.smoothing_length_D*rotor_diameter_i,
            self.mixing_gain_velocity*mixing_i,
        )
        sigma_y[upstream_mask] = sigma_y0.flatten()[0]
        
        sigma_z = geometric_model_wake_width(
            x-x_i, 
            self.wake_expansion_rates, 
            [b*rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_z0, 
            self.smoothing_length_D*rotor_diameter_i,
            self.mixing_gain_velocity*mixing_i,
        )
        sigma_z[upstream_mask] = sigma_z0.flatten()[0]
        
        # 'Standard' wake component
        r, C = rCalt(
            wind_veer,
            sigma_y,
            sigma_z,
            y,
            y_i,
            deflection_field_y_i,
            deflection_field_z_i,
            z,
            hub_height_i,
            ct_i,
            yaw_angle,
            tilt_angle_i,
            rotor_diameter_i,
            sigma_y0,
            sigma_z0
        )
        # Normalize to match end of acuator disk model tube
        C = C / (8*(self.sigma_y0_D**2))
        
        wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
        
        if include_mirror_wake:
            # TODO: speed up this option by calculating various elements in 
            #       rCalt only once.
            # Mirror component
            r_mirr, C_mirr = rCalt(
                wind_veer, # TODO: Is veer OK with mirror wakes?
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_y_i,
                deflection_field_z_i,
                z,
                -hub_height_i, # Turbine at negative hub height location
                ct_i,
                yaw_angle,
                tilt_angle_i,
                rotor_diameter_i,
                sigma_y0,
                sigma_z0
            )
            # Normalize to match end of acuator disk model tube
            C_mirr = C_mirr / (8*(self.sigma_y0_D**2))
            
            # ASSUME sum-of-squares superposition for the real and mirror wakes
            wake_deficit = np.sqrt(
                wake_deficit**2 + 
                gaussian_function(C_mirr, r_mirr, 1, np.sqrt(0.5))**2
            )

        velocity_deficit = wake_deficit * downstream_mask

        return velocity_deficit


# @profile
def rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D):

    ## original
    # a = cosd(wind_veer) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer) ** 2 / (2 * sigma_z ** 2)
    # b = -sind(2 * wind_veer) / (4 * sigma_y ** 2) + sind(2 * wind_veer) / (4 * sigma_z ** 2)
    # c = sind(wind_veer) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer) ** 2 / (2 * sigma_z ** 2)
    # r = (
    #     a * (y - y_i - delta) ** 2
    #     - 2 * b * (y - y_i - delta) * (z - HH)
    #     + c * (z - HH) ** 2
    # )
    # C = 1 - np.sqrt(np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0))

    ## Precalculate some parts
    # twox_sigmay_2 = 2 * sigma_y ** 2
    # twox_sigmaz_2 = 2 * sigma_z ** 2
    # a = cosd(wind_veer) ** 2 / (twox_sigmay_2) + sind(wind_veer) ** 2 / (twox_sigmaz_2)
    # b = -sind(2 * wind_veer) / (2 * twox_sigmay_2) + sind(2 * wind_veer) / (2 * twox_sigmaz_2)
    # c = sind(wind_veer) ** 2 / (twox_sigmay_2) + cosd(wind_veer) ** 2 / (twox_sigmaz_2)
    # delta_y = y - y_i - delta
    # delta_z = z - HH
    # r = (a * (delta_y ** 2) - 2 * b * (delta_y) * (delta_z) + c * (delta_z ** 2))
    # C = 1 - np.sqrt(np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / (D * D))), 0.0, 1.0))

    ## Numexpr
    wind_veer = np.deg2rad(wind_veer)
    a = ne.evaluate(
        "cos(wind_veer) ** 2 / (2 * sigma_y ** 2) + sin(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    b = ne.evaluate(
        "-sin(2 * wind_veer) / (4 * sigma_y ** 2) + sin(2 * wind_veer) / (4 * sigma_z ** 2)"
    )
    c = ne.evaluate(
        "sin(wind_veer) ** 2 / (2 * sigma_y ** 2) + cos(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    r = ne.evaluate(
        "a * ((y - y_i - delta) ** 2) - 2 * b * (y - y_i - delta) * (z - HH) + c * ((z - HH) ** 2)"
    )
    d = np.clip(1 - (Ct * cosd(yaw) / ( 8.0 * sigma_y * sigma_z / (D * D) )), 0.0, 1.0)
    C = ne.evaluate("1 - sqrt(d)")
    return r, C

def rCalt(wind_veer, sigma_y, sigma_z, y, y_i, delta_y, delta_z, z, HH, Ct, yaw, tilt, D, sigma_y0, sigma_z0):

    ## Numexpr
    wind_veer = np.deg2rad(wind_veer)
    a = ne.evaluate("cos(wind_veer) ** 2 / (2 * sigma_y ** 2) + sin(wind_veer) ** 2 / (2 * sigma_z ** 2)")
    b = ne.evaluate("-sin(2 * wind_veer) / (4 * sigma_y ** 2) + sin(2 * wind_veer) / (4 * sigma_z ** 2)")
    c = ne.evaluate("sin(wind_veer) ** 2 / (2 * sigma_y ** 2) + cos(wind_veer) ** 2 / (2 * sigma_z ** 2)")
    r = ne.evaluate("a * ( (y - y_i - delta_y) ** 2) - 2 * b * (y - y_i - delta_y) * (z - HH - delta_z) + c * ((z - HH - delta_z) ** 2)")
    d = 1 - Ct * (sigma_y0 * sigma_z0)/(sigma_y * sigma_z) * cosd(yaw) * cosd(tilt)
    C = ne.evaluate("1 - sqrt(d)")
    return r, C

def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR

def gaussian_function(C, r, n, sigma):
    return C * np.exp(-1 * r ** n / (2 * sigma ** 2))

def sigmoid_integral(x, center=0, width=1):
    w = width/(2*np.log(0.95/0.05))
    return w*np.log(np.exp((x-center)/w) + 1)

def geometric_model_wake_width(
    x, 
    wake_expansion_rates, 
    breakpoints, 
    sigma_0, 
    smoothing_length,
    mixing_final,
    ):
    assert len(wake_expansion_rates) == len(breakpoints) + 1, \
        "Invalid combination of wake_expansion_rates and breakpoints."

    sigma = (wake_expansion_rates[0] + mixing_final) * x + sigma_0
    for ib, b in enumerate(breakpoints):
        sigma += (wake_expansion_rates[ib+1]-wake_expansion_rates[ib]) * \
            sigmoid_integral(x, center=b, width=smoothing_length) 

    return sigma