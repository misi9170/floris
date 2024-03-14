import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def toy_ode_system(t, y):
    return [y[1], -np.sin(y[0])]


def compute_centerline_velocities(x_, U_inf):
    """
    Compute the centerline velocities using the eddy viscosity model
    x_ supposed to be defined from the center of the rotor
    (0 at the rotor location).
    """
    U_c = 0
    return U_c


def compute_off_center_velocities(U_c, y_, z_, Ct):
    """
    Compute the off-centerline velocities using the eddy viscosity model
    y_, z_ supposed to be defined from the center of the rotor.
    """
    
    w_sq = wake_width_squared(Ct, U_c)
    U_r = 1 - (1-U_c) * np.exp(-(y_**2 + z_**2)/w_sq)
    return U_r

def wake_width_squared(Ct, U_c):
    """
    Compute the wake width squared using the eddy viscosity model
    """
    return Ct / (4*(1-U_c)*(1+U_c))

def centerline_ode(x_, U_c_, U_inf, ambient_ti, Ct, hh, D):
    """
    Define the ODE for the centerline velocities
    """
    # Define constants (will later define these as class attribtues)
    k_l = 0.015*np.sqrt(3.56)
    k_a = 0.5
    # ambient_ti = 0.06
    #U_inf = 8.0 # Will be passed in as an argument
    #hh = 90.0 # Will be passed in as an argument
    #Ct = 0.9 # Will be passed in as an argument
    von_Karman = 0.41
    
    length_scale = von_Karman*hh

    K_l = k_l * np.sqrt(wake_width_squared(Ct, U_c_)) * (U_inf - U_c_*U_inf) # local component
    K_a = k_a * ambient_ti * U_inf * length_scale # ambient component (9)

    def filter_function(x_):
        """ Identity mapping (assumed by 'F=1') """

        # Wait, is this just a multiplier? Seems to be?
        filter_const_1 = 0.65
        filter_const_2 = 4.5
        filter_const_3 = 23.32
        filter_const_4 = 1/3
        filter_cutoff_x_ = 5.5
        if x_ < filter_cutoff_x_: # How should this work? Is this smooth??
            return filter_const_1 * ((x_ - filter_const_2) / filter_const_3)**filter_const_4
        else:
            return x_

    eddy_viscosity = filter_function(K_l + K_a)
    ev_ = eddy_viscosity/(U_inf*D)

    dU_c__dx_ = 16 * ev_ * (U_c_**3 - U_c_**2 - U_c_ + 1) / (U_c_ * Ct)
        
    return [dU_c__dx_]

def initial_U_c_(Ct, ambient_ti):

    i_const_1 = 0.05
    i_const_2 = 16
    i_const_3 = 0.5

    # The below are from Ainslie (1988)
    initial_vel_def = Ct - i_const_1 - (i_const_2 * Ct - i_const_3) * ambient_ti / 1000

    U_c0_ = 1 - initial_vel_def

    return U_c0_



if __name__ == "__main__":

    # 0 to 20 diameters
    x__span = [0, 20] # 

    Ct = 0.8
    hh = 90.0
    D = 126.0
    ambient_ti = 0.06
    U_inf = 8.0

    U_c0_ = initial_U_c_(Ct, ambient_ti)

    # Solve the ODE
    sol = solve_ivp(
        centerline_ode,
        x__span,
        [U_c0_],
        method='RK45',
        args=(U_inf, ambient_ti, Ct, hh, D)
    )

    # Extract the solution
    x__out = sol.t
    U_c__out = sol.y.flatten()

    fig, ax = plt.subplots(2,1)
    ax[0].plot(x__out, U_c__out)
    ax[0].set_xlabel("x [D]")
    ax[0].set_ylabel("U_c_ [-]")

    ax[1].plot(x__out*D, U_c__out*U_inf)
    ax[1].plot(x__out*D, np.ones_like(x__out)*U_inf)
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("U_c [m/s]")
    
    plt.show()

