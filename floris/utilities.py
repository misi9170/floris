
from __future__ import annotations

import os
from math import ceil
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import yaml
from attrs import define, field
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import odeint

from floris.type_dec import floris_array_converter, NDArrayFloat


def pshape(array: np.ndarray, label: str = ""):
    print(label, np.shape(array))


def cosd(angle):
    """
    Cosine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.cos(np.radians(angle))


def sind(angle):
    """
    Sine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.sin(np.radians(angle))


def tand(angle):
    """
    Tangent of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.tan(np.radians(angle))


def wrap_180(x):
    """
    Shift the given values to within the range (-180, 180].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.ndarray | float | int: Shifted values.
    """

    return ((x + 180.0) % 360.0) - 180.0


def wrap_360(x):
    """
    Shift the given values to within the range (0, 360].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.ndarray | float | int: Shifted values.
    """

    return x % 360.0


def wind_delta(wind_directions: NDArrayFloat | float):
    """
    This function calculates the deviation from West (270) for a given wind direction or series
    of wind directions. First, 270 is subtracted from the input wind direction, and then the
    remainder after dividing by 360 is retained (modulo). The table below lists examples of
    results.

    | Input | Result |
    | ----- | ------ |
    | 270.0 | 0.0    |
    | 280.0 | 10.0   |
    | 360.0 | 90.0   |
    | 180.0 | 270.0  |
    | -10.0 | 80.0   |
    |-100.0 | 350.0  |

    Args:
        wind_directions (NDArrayFloat | float): A single or series of wind directions. They can be
        any number, negative or positive, but it is typically between 0 and 360.

    Returns:
        NDArrayFloat | float: The delta between the given wind direction and 270 in positive
        quantities between 0 and 360. The returned type is the same as wind_directions.
    """

    return (wind_directions - 270) % 360


def rotate_coordinates_rel_west(
    wind_directions,
    coordinates,
    x_center_of_rotation=None,
    y_center_of_rotation=None
):
    """
    This function rotates the given coordinates so that they are aligned with West (270) rather
    than North (0). The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        coordinates (NDArrayFloat): Series of coordinates to rotate with shape (N coordinates, 3)
            so that each element of the array coordinates[i] yields a three-component coordinate.
        x_center_of_rotation (float, optional): The x-coordinate for the rotation center of the
            input coordinates. Defaults to None.
        y_center_of_rotation (float, optional): The y-coordinate for the rotational center of the
            input coordinates. Defaults to None.
    """

    # Calculate the difference in given wind direction from 270 / West
    wind_deviation_from_west = wind_delta(wind_directions)
    wind_deviation_from_west = np.expand_dims(wind_deviation_from_west, axis=-1)

    # Construct the arrays storing the turbine locations
    x_coordinates, y_coordinates, z_coordinates = coordinates.T

    # Find center of rotation - this is the center of box bounding all of the turbines
    if x_center_of_rotation is None:
        x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
    if y_center_of_rotation is None:
        y_center_of_rotation = (np.min(y_coordinates) + np.max(y_coordinates)) / 2

    # Rotate turbine coordinates about the center
    x_coord_offset = x_coordinates - x_center_of_rotation
    y_coord_offset = y_coordinates - y_center_of_rotation
    x_coord_rotated = (
        x_coord_offset * cosd(wind_deviation_from_west)
        - y_coord_offset * sind(wind_deviation_from_west)
        + x_center_of_rotation
    )
    y_coord_rotated = (
        x_coord_offset * sind(wind_deviation_from_west)
        + y_coord_offset * cosd(wind_deviation_from_west)
        + y_center_of_rotation
    )
    z_coord_rotated = np.ones_like(wind_deviation_from_west) * z_coordinates
    return x_coord_rotated, y_coord_rotated, z_coord_rotated, x_center_of_rotation, \
        y_center_of_rotation


def reverse_rotate_coordinates_rel_west(
    wind_directions: NDArrayFloat,
    grid_x: NDArrayFloat,
    grid_y: NDArrayFloat,
    grid_z: NDArrayFloat,
    x_center_of_rotation: float,
    y_center_of_rotation: float
):
    """
    This function reverses the rotation of the given grid so that the coordinates are aligned with
    the original wind direction. The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        grid_x (NDArrayFloat): X-coordinates to be rotated.
        grid_y (NDArrayFloat): Y-coordinates to be rotated.
        grid_z (NDArrayFloat): Z-coordinates to be rotated.
        x_center_of_rotation (float): The x-coordinate for the rotation center of the
            input coordinates.
        y_center_of_rotation (float): The y-coordinate for the rotational center of the
            input coordinates.
    """
    # Calculate the difference in given wind direction from 270 / West
    # We are rotating in the other direction
    wind_deviation_from_west = -1.0 * wind_delta(wind_directions)

    # Construct the arrays storing the turbine locations
    grid_x_reversed = np.zeros_like(grid_x)
    grid_y_reversed = np.zeros_like(grid_x)
    grid_z_reversed = np.zeros_like(grid_x)
    for wii, angle_rotation in enumerate(wind_deviation_from_west):
        x_rot = grid_x[wii]
        y_rot = grid_y[wii]
        z_rot = grid_z[wii]

        # Rotate turbine coordinates about the center
        x_rot_offset = x_rot - x_center_of_rotation
        y_rot_offset = y_rot - y_center_of_rotation
        x = (
            x_rot_offset * cosd(angle_rotation)
            - y_rot_offset * sind(angle_rotation)
            + x_center_of_rotation
        )
        y = (
            x_rot_offset * sind(angle_rotation)
            + y_rot_offset * cosd(angle_rotation)
            + y_center_of_rotation
        )
        z = z_rot  # Nothing changed in this rotation

        grid_x_reversed[wii] = x
        grid_y_reversed[wii] = y
        grid_z_reversed[wii] = z

    return grid_x_reversed, grid_y_reversed, grid_z_reversed

def warp_grid_for_wind_direction_heterogeneity(
        x,
        y,
        z,
        wd_het_x_points,
        wd_het_values,
    ):

    y_warped = y.copy()

    for i in range(wd_het_values.shape[1]-1):

        x0 = wd_het_x_points[:,i:i+1]
        x1 = wd_het_x_points[:,i+1:i+2]

        # Compute radius of curvature for the warp (not sign swap in tand)
        warp_radius = (x1 - x0) / tand(wd_het_values[:,i:i+1] - wd_het_values[:,i+1:i+2])

        ### Below is code for a "true" warp in x and y, but is not complete.
        # # Compute the angle of warping for each grid point (radians)
        # warp_theta = np.arctan2(x - x0, np.abs(warp_radius))*np.sign(warp_radius)

        # # Apply warp
        # import ipdb; ipdb.set_trace()
        # x_warped = np.where( # Also add end points
        #     (x >= x0) & (x <= x1) & (np.abs(warp_radius) < np.inf),
        #     warp_radius * np.sin(warp_theta) + y * np.sin(warp_theta) + x0,
        #     x
        # )
        # y_warped = np.where(
        #     (x >= x0) & (x <= x1) & (np.abs(warp_radius) < np.inf),
        #     warp_radius * np.cos(warp_theta) + y * np.cos(warp_theta) - warp_radius,
        #     y
        # )

        ### Instead, use a simpler y "slide" approach
        if len(x.shape) == len(x1.shape):
            x1_rep = np.repeat(x1, x.shape[1], axis=1) # Usual solve
        else:
            x1_rep = x1[:,:,None] # Full field solve
        x_mod = np.where(
            x > x1_rep,
            x1_rep,
            x
        )
        y_warped = np.where(
            (np.abs(warp_radius)  != np.inf ) & (x >= x0),
            (-warp_radius + np.sign(warp_radius) * np.sqrt(warp_radius**2 - (x_mod - x0)**2)
             + y_warped),
            y_warped
        )

    # Return warped grid
    return x, y_warped, z

def apply_wind_direction_heterogeneity(
        x,
        y,
        z,
        wind_direction_x_points,
        wind_direction_y_points,
        wind_direction_z_points,
        wind_direction_u_values,
        wind_direction_v_values,
        wind_direction_w_values,
        n_turbines
    ):

    compute_streamline = compute_streamline_2D# if wind_direction_z_points is None else compute_streamline_3D
    shift_points_by_streamline = shift_points_by_streamline_2D# if wind_direction_z_points is None else shift_points_by_streamline_3D

    # Initialize storage
    x_per_turbine = np.repeat(x[:,:,:,:,None], n_turbines, axis=4)
    y_per_turbine = np.repeat(y[:,:,:,:,None], n_turbines, axis=4)
    z_per_turbine = np.repeat(z[:,:,:,:,None], n_turbines, axis=4)

    n_findex = x.shape[0]

    # Ensure all needed elements are arrays
    wind_direction_x_points = np.array(wind_direction_x_points)
    wind_direction_y_points = np.array(wind_direction_y_points)
    wind_direction_u_values = np.array(wind_direction_u_values)
    wind_direction_v_values = np.array(wind_direction_v_values)
    if wind_direction_z_points is not None:
        wind_direction_z_points = np.array(wind_direction_z_points)
    if wind_direction_w_values is not None:
        wind_direction_w_values = np.array(wind_direction_w_values)

    # Compute new relative locations
    # TEMP________
    option = 3
    if option == 1:
        # Place turbine 2 directly downstream of turbine 1, as seen by turbine 1
        y_per_turbine[0,2,:,:,1] = y_per_turbine[0,1,:,:,1]
        # Place turbine 3 out of the direct path of turbine 1, as seen by turbine 1
        y_per_turbine[0,3,:,:,1] = 2*y_per_turbine[0,3,:,:,1] 
    elif option == 2:
        #y_per_turbine[0,:,:,:,0] = y_per_turbine[0,:,:,:,0] - 0.2*x_per_turbine[0,:,:,:,0]
        #y_per_turbine[0,:,:,:,1] = y_per_turbine[0,:,:,:,1] + 0.2*x_per_turbine[0,:,:,:,1]
        y_per_turbine[0,:,:,:,0] = y_per_turbine[0,:,:,:,0] + 0.2*x_per_turbine[0,:,:,:,0]
    elif option == 3:
        # TODO: compute streamlines; convert to point movements.
        # TODO: Can I avoid any of these looping operations? Do I need to?
        for findex in range(n_findex):
            for tindex in range(n_turbines):
                # 1. Compute the streamline
                streamline = compute_streamline(
                    wind_direction_x_points,
                    wind_direction_y_points,
                    None if wind_direction_z_points is None else wind_direction_z_points[findex,:],
                    wind_direction_u_values[findex,:],
                    wind_direction_v_values[findex,:],
                    None if wind_direction_w_values is None else wind_direction_w_values[findex,:],
                    np.array([
                        x[findex,tindex,:,:].mean(),
                        y[findex,tindex,:,:].mean(),
                        z[findex,tindex,:,:].mean()
                    ])
                )

                # 2. Compute the point movements
                x_shifted, y_shifted, z_shifted = shift_points_by_streamline(
                    streamline, x[findex,:,:,:], y[findex,:,:,:], z[findex,:,:,:]
                )
                
                # 3. Apply to per_turbine values
                x_per_turbine[findex,:,:,:,tindex] = x_shifted
                y_per_turbine[findex,:,:,:,tindex] = y_shifted
                z_per_turbine[findex,:,:,:,tindex] = z_shifted
    else:
        pass # make no change

    # END TEMP____

    # Return full set of locations
    return x_per_turbine, y_per_turbine, z_per_turbine

def compute_streamline_3D(x, y, z, u, v, w, xyz_0):
    """
    Compute streamline starting at xyz_0.
    """
    s = np.linspace(0, 1, 1000) # parametric variable

    scale = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
    offset = np.array([(x.max() + x.min())/2, (y.max() + y.min())/2, (z.max() + z.min())/2])
    # Need to compute the offset too

    # Collect
    xyz = np.array([x, y, z]).T
    uvw = np.concatenate((u, v, w), axis=0).T

    # Normalize
    xyz = (xyz - offset) / scale
    xyz_0 = (xyz_0 - offset) / scale
    # TODO: Loop over findices! Maybe not necessary---can I compute all streamlines at once??
    # Would it help? Currently assuming a single findex.
    interp = LinearNDInterpolator(xyz, uvw, fill_value=0.0)

    def velocity_field_interpolate_reg(xyz_, s):
        # if (xyz_ <= -0.5).any() or (xyz_ >= 0.5).any():
        #     print("hmm")
        #     return np.array([0.0, 0.0, 0.0])
        # else:
        #     return interp(xyz_)[0]
        return interp(xyz_)[0]

    streamline = odeint(velocity_field_interpolate_reg, xyz_0, s) # 1000 x 3

    # Denormalize
    streamline = streamline * scale + offset

    return streamline

def compute_streamline_2D(x, y, z, u, v, w, xyz_0, n_points=1000):
    """
    Compute streamline starting at xyz_0.
    z and w will be ignored.
    """

    xy_0 = xyz_0[:2]

    s = np.linspace(0, 1, n_points) # parametric variable

    scale = np.array([x.max() - x.min(), y.max() - y.min()])
    offset = np.array([(x.max() + x.min())/2, (y.max() + y.min())/2])
    # Need to compute the offset too

    # Collect
    xy = np.array([x, y]).T
    uv = np.array([u,v]).T

    # Normalize
    xy = (xy - offset) / scale
    xy_0 = (xy_0 - offset) / scale
    # TODO: Loop over findices! Maybe not necessary---can I compute all streamlines at once??
    # Would it help? Currently assuming a single findex.

    interp = LinearNDInterpolator(xy, uv)

    def velocity_field_interpolate_reg(xy_, s):
        if (xy_ < -0.5).any() or (xy_ > 0.5).any():
            return np.array([0, 0])
        else:
            return interp(xy_)[0]

    streamline = odeint(velocity_field_interpolate_reg, xy_0, s) # n_points x 2

    # Determine point of exit from specified domain
    if (streamline < -0.5).any():
        low_exit = np.argwhere(streamline < -0.5)[0,0]
    else:
        low_exit = 1000
    if (streamline > 0.5).any():
        high_exit = np.argwhere(streamline > 0.5)[0,0]
    else:
        high_exit = 1000
    exit_index = np.minimum(low_exit, high_exit)
    if exit_index == 0:
        raise ValueError("Streamline could not be propagated inside the domain.")
    if exit_index == 1000:
        raise ValueError("Streamline did not reach the edge of the domain.")
    
    # Copy points after exit
    streamline[exit_index:] = streamline[exit_index-1]

    # Denormalize
    streamline = streamline * scale + offset

    return streamline

def shift_points_by_streamline_2D(streamline, x, y, z):
    ## I think should be dist by turbine only (not each rotor point); and then 
    # shift all rotor points by the same amount.
    # TODO: How will this work over findices? Need to work that out.
    xy = np.concatenate((x.mean(axis=(1,2))[None,:],y.mean(axis=(1,2))[None,:]), axis=0)

    rotor_y = y - y.mean(axis=(1,2), keepdims=True)
    rotor_z = z - z.mean(axis=(1,2), keepdims=True)

    # Storage for new points
    x_new = x.copy()
    y_new = y.copy()

    disps_to_streamline = streamline[:,:,None] - xy[None,:, :]
    dists_to_streamline = np.linalg.norm(disps_to_streamline, axis=1)
    min_idx = np.argmin(dists_to_streamline, axis=0)
    #points = streamline[min_idx, :]

    # At this point, loop over the (downstream) turbines for the shifts
    # TODO: for viz/solve for points, are there many more points than turbines?
    # If so, maybe need to work on a non-looped version of this.
    for tindex in range(len(min_idx)):
        streamline_diffs = np.diff(streamline[:min_idx[tindex], :], axis=0)
        along_stream_dist = np.sum(np.linalg.norm(streamline_diffs, axis=1))
        off_stream_dist = dists_to_streamline[min_idx[tindex], tindex]
        #off_stream_disp = disps_to_streamline[min_idx[tindex], :, tindex]
        #disp_x = off_stream_disp[0]
        #disp_y = off_stream_disp[1]
        
        #theta = np.arctan2(disp_y, disp_x)
        
        x_new[tindex,:,:] = streamline[0,0] + along_stream_dist
        #y_new[:,tindex,:,:] = streamline[0,1] + disp_x*np.sin(theta) + disp_y*np.cos(theta) + rotor_y[:,tindex,:,:]
        y_new[tindex,:,:] = streamline[0,1] + off_stream_dist + rotor_y[tindex,:,:]

    return x_new, y_new, z

class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)


Loader.add_constructor('!include', Loader.include)

def load_yaml(filename, loader=Loader):
    with open(filename) as fid:
        return yaml.load(fid, loader)


def round_nearest_2_or_5(x: int | float) -> int:
    """Rounds a number (with a 0.5 buffer) up to the nearest integer divisible by 2 or 5.

    Args:
        x (int | float): The number to be rounded.

    Returns:
        int: The rounded number.
    """
    base_2 = 2
    base_5 = 5
    return min(base_2 * ceil((x + 0.5) / base_2), base_5 * ceil((x + 0.5) / base_5))


def round_nearest(x: int | float, base: int = 5) -> int:
    """Rounds a number (with a 0.5 buffer) up to the nearest integer divisible by 5.

    Args:
        x (int | float): The number to be rounded.

    Returns:
        int: The rounded number.
    """
    return base * ceil((x + 0.5) / base)


def nested_get(
    d: Dict[str, Any],
    keys: List[str]
) -> Any:
    """Get a value from a nested dictionary using a list of keys.
    Based on:
    https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        d (Dict[str, Any]): The dictionary to get the value from.
        keys (List[str]): A list of keys to traverse the dictionary.

    Returns:
        Any: The value at the end of the key traversal.
    """
    for key in keys:
        d = d[key]
    return d

def nested_set(
    d: Dict[str, Any],
    keys: List[str],
    value: Any,
    idx: Optional[int] = None
) -> None:
    """Set a value in a nested dictionary using a list of keys.
    Based on:
    https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to set the value in.
        keys (List[str]): A list of keys to traverse the dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): If the value is an list, the index to change.
            Defaults to None.
    """
    d_in = d.copy()

    for key in keys[:-1]:
        d = d.setdefault(key, {})
    if idx is None:
        # Parameter is a scalar, set directly
        d[keys[-1]] = value
    else:
        # Parameter is a list, need to first get the list, change the values at idx

        # # Get the underlying list
        par_list = nested_get(d_in, keys)
        par_list[idx] = value
        d[keys[-1]] = par_list

def print_nested_dict(dictionary: Dict[str, Any], indent: int = 0) -> None:
    """Print a nested dictionary with indentation.

    Args:
        dictionary (Dict[str, Any]): The dictionary to print.
        indent (int, optional): The number of spaces to indent. Defaults to 0.
    """
    for key, value in dictionary.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent + 4)
        else:
            print(" " * (indent + 4) + str(value))

def update_model_args(
        model_args: Dict[str, Any],
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ):
    """
    Update the x_sorted, y_sorted, and z_sorted entries of the model_args dict.
    """
    # Check that new x, y, z are the same length as the old
    if (x.shape != model_args["x"].shape or 
        y.shape != model_args["y"].shape or
        z.shape != model_args["z"].shape):
        raise ValueError("Size of new x,y,z arrays must match existing ones.")
    
    # Update the x, y, z values
    model_args["x"] = x
    model_args["y"] = y
    model_args["z"] = z
    
    return model_args
