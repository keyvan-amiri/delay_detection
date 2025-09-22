# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:15:50 2025
@author: kamirel
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

def phi_extremes(y, extr_type="high", coef=1.5, asym=True):
    y = np.asarray(y)
    q1, q2, q3 = np.percentile(y, [25, 50, 75])
    iqr = q3 - q1
    if asym:
        # adjusted fences (approx robustbase::adjboxStats)
        lower_fence = q1 - coef * iqr * 1.5
        upper_fence = q3 + coef * iqr * 1.5
    else:
        lower_fence = q1 - coef * iqr
        upper_fence = q3 + coef * iqr

    rmin, rmax = np.min(y), np.max(y)

    control_pts = []
    if extr_type in ("both","low"):
        control_pts.append([lower_fence, 1, 0])
    else:
        control_pts.append([rmin, 0, 0])

    control_pts.append([q2, 0, 0])  # median

    if extr_type in ("both","high"):
        control_pts.append([upper_fence, 1, 0])
    else:
        control_pts.append([rmax, 0, 0])

    control_pts = np.array(control_pts)
    return {"method":"extremes", "npts":len(control_pts), "control.pts":control_pts}

def phi_range(y, control_pts):
    control_pts = np.asarray(control_pts, dtype=float)
    if control_pts.shape[1] == 2:
        # estimate slopes if not given
        dx = np.diff(control_pts[:,0])
        dy = np.diff(control_pts[:,1])
        slopes = np.r_[0, (dy[1:]/dx[1:] + dy[:-1]/dx[:-1])/2, 0]
        control_pts = np.c_[control_pts, slopes]
    control_pts = control_pts[np.argsort(control_pts[:,0])]
    return {"method":"range", "npts":len(control_pts), "control.pts":control_pts}

def phi_control(y, method="extremes", extr_type="both", control_pts=None, asym=True):
    if method == "extremes":
        return phi_extremes(y, extr_type=extr_type, asym=asym)
    elif method == "range":
        return phi_range(y, control_pts)
    else:
        raise ValueError("method must be 'extremes' or 'range'")

def phi(y, phi_parms):
    y = np.asarray(y)
    cpts = np.asarray(phi_parms["control.pts"])
    xs, ys = cpts[:,0], cpts[:,1]
    interp = PchipInterpolator(xs, ys, extrapolate=True)
    return np.clip(interp(y), 0, 1)