import numpy as np
from scipy import optimize

__all__ = ['circle_fit', 'cart2pol']


def circle_fit(x, y):
    ###
    # this function fit a circle given (x,y) scatter points in a plane.
    #
    # INPUT:
    #
    #   x................numpy array (n,) where n is the length of the array
    #   y................numpy array (n,) where n is the length of the array
    #
    # OUTPUT:
    #
    #   xc_2.............scalar, it is the coordinate x of the fitted circle
    #   yc_2.............scalar, it is the coordinate y of the fitted circle
    #   R_2..............scalar, it is the radius of the fitted circle
    ###

    # initialize the coordinate for the fitting procedure
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(xc_2, yc_2)
    R_2 = Ri_2.mean()
    # residu_2   = sum((Ri_2 - R_2)**2)
    # residu2_2  = sum((Ri_2**2-R_2**2)**2)

    return xc_2, yc_2, R_2


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    phi = phi * (180 / np.pi)  # degrees
    phi[phi == 180] = -180

    return rho, phi
