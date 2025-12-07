import numpy as np

MOON_G = np.array([0.0, 0.0, -1.62])

def ease_in_out(u):
    u = np.clip(u, 0.0, 1.0)
    s = 0.5 - 0.5*np.cos(np.pi*u)
    ds_dt  = (np.pi)*np.sin(np.pi*u)      # 后面会除以 T，总体形状即可
    d2s_dt2 = (np.pi**2)*np.cos(np.pi*u)
    return s, ds_dt, d2s_dt2

def B_quad(p0, p1, p2, s):
    return (1-s)**2*p0 + 2*(1-s)*s*p1 + s**2*p2

def dB_quad(p0, p1, p2, s):
    return 2*(1-s)*(p1-p0) + 2*s*(p2-p1)

def d2B_quad(p0, p1, p2):
    return 2*(p2 - 2*p1 + p0)

def path_ref(t, t_final, pA, pB, mode="landing", curve_apex_z=35.0):
    u = t / t_final
    s, ds_dt, d2s_dt2 = ease_in_out(u)

    if mode == "landing":
        C = np.array([pB[0], pB[1], max(pA[2], pB[2]) + curve_apex_z])
    else:
        C = (pA + pB)/2

    p_ref = B_quad(pA, C, pB, s)
    dBds  = dB_quad(pA, C, pB, s)
    d2Bds2 = d2B_quad(pA, C, pB)

    v_ref = dBds * ds_dt
    a_ff  = d2Bds2*(ds_dt**2) + dBds*d2s_dt2
    return p_ref, v_ref, a_ff

def angles_from_dir(tB):
    tx, ty, tz = tB
    denom = np.sqrt(max(0.0, ty*ty + tz*tz))
    theta_y = np.arctan2(tx, denom)      # yaw around y
    theta_p = np.arctan2(-ty, tz)        # pitch around x
    return theta_p, theta_y
