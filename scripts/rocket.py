from dataclasses import dataclass, field
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------- Utilities -----------------

@dataclass
class Params:
    g: np.ndarray
    rho: float
    CdA: float
    Isp: float
    g0: float
    Tmin: float
    Tmax: float
    vz_cmd: float = 20.0
    k_vz: float = 2.0
    k_az_smooth: float = 0.7
    t_ign: float = 0.10
    tilt_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    liftoff_margin: float = 0.05
    theta_gim: float = np.deg2rad(15.0)
    v_touch: float = 0.75
    pad_r: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    gs_enable: bool = True
    gs_angle: float = np.deg2rad(75.0)
    gs_k: float = 0.6
    k_pos: float = 0.9
    k_vel: float = 1.8
    tgo_min: float = 2.5


def vecnorm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(v: np.ndarray) -> np.ndarray:
    n = vecnorm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


# ----------------- 1) Liftoff dynamics + controllers -----------------

def thrust_controller_liftoff(t: float, x: np.ndarray, P: Params, az_filt_cache: dict) -> np.ndarray:
    """
    Vertical speed P-controller with simple smoothing, throttle limits, and liftoff margin.
    Returns thrust vector in world frame (here constrained to +z only for simplicity).
    """
    v = x[3:6]
    m = x[6]

    # No thrust before ignition time
    if t < P.t_ign:
        return np.zeros(3)

    # Exponential smoothing on commanded vertical acceleration
    evz = P.vz_cmd - v[2]
    az_cmd = P.k_vz * evz
    az_prev = az_filt_cache.get("az", 0.0)
    az_filt = (1.0 - P.k_az_smooth) * az_prev + P.k_az_smooth * az_cmd
    az_filt_cache["az"] = az_filt

    # Required thrust magnitude to cancel gravity plus commanded acceleration
    # Note: P.g[2] < 0, so subtracting gives +mg compensation
    T_mag_cmd = m * (az_filt - P.g[2])
    T_mag = np.clip(T_mag_cmd, P.Tmin, P.Tmax)

    # If still on the ground, enforce a margin above hover thrust to push off the pad
    if x[2] <= 1e-6:
        T_hover = max(1e-6, m * P.g0)
        T_mag = max(T_mag, (1.0 + P.liftoff_margin) * T_hover)

    # Pure vertical thrust (no tilt in this simple liftoff demo)
    return T_mag * np.array([0.0, 0.0, 1.0])


def dyn_with_controller_and_ground(t: float, x: np.ndarray, P: Params, az_filt_cache: dict) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    m = x[6]

    # Drag opposite to velocity
    if P.rho > 0 and P.CdA > 0 and vecnorm(v) > 1e-6:
        vhat = v / vecnorm(v)
        q = 0.5 * P.rho * (vecnorm(v) ** 2)
        aD = -(q * P.CdA / max(m, 1e-6)) * vhat
    else:
        aD = np.zeros(3)

    # Thrust from the controller
    T = thrust_controller_liftoff(t, x, P, az_filt_cache)

    # Ground contact model: if on ground and thrust not enough to liftoff, z-motion locked
    on_ground = (r[2] <= 0.0) and (v[2] <= 0.0)
    T_weight = m * P.g0
    mustBeat = (1.0 + P.liftoff_margin) * T_weight

    if on_ground and vecnorm(T) <= mustBeat:
        rdot = np.array([v[0], v[1], 0.0])
        vdot = np.array([P.g[0] + aD[0], P.g[1] + aD[1], 0.0])  # lock vertical motion
        mdot = -vecnorm(T) / (P.Isp * P.g0)                      # allow fuel consumption during ignition
        return np.hstack([rdot, vdot, mdot])

    # Numerical guard: do not sink below the ground after liftoff
    if r[2] < 0.0:
        r = r.copy(); v = v.copy()
        r[2] = 0.0
        v[2] = max(v[2], 0.0)

    # Free flight
    rdot = v
    vdot = P.g + aD + T / max(m, 1e-6)
    mdot = -vecnorm(T) / (P.Isp * P.g0)
    return np.hstack([rdot, vdot, mdot])


def event_fuel_out_liftoff(_t: float, x: np.ndarray, *_args) -> float:
    """Terminate when mass <= 100 kg."""
    return x[6] - 100.0


# ----------------- 2) Step visualization -----------------

def run_rocket_liftoff_stepviz():
    # Physical params (Earth-like)
    P = Params(
        g=np.array([0.0, 0.0, -9.80665]),
        rho=1.20,
        CdA=0.5 * 1.0,
        Isp=280.0,
        g0=9.80665,
        Tmin=0.10 * 1.5e5,
        Tmax=1.5e5,
    )
    r0 = np.array([0.0, 0.0, 0.0])  # on the ground
    v0 = np.array([0.0, 0.0, 0.0])
    m0 = 1500.0
    x0 = np.hstack([r0, v0, m0])

    tspan = (0.0, 60.0)
    az_filt_cache = {"az": 0.0}
    fun = lambda t, x: dyn_with_controller_and_ground(t, x, P, az_filt_cache)

    sol = solve_ivp(
        fun, tspan, x0, method="RK45",
        rtol=1e-7, atol=1e-8,
        events=[event_fuel_out_liftoff],
        dense_output=False, max_step=0.02
    )
    t = sol.t
    X = sol.y.T  # shape (N,7)

    # Recompute thrust for plotting
    Tcmd = np.zeros((len(t), 3))
    az_filt_cache = {"az": 0.0}
    for k in range(len(t)):
        Tcmd[k, :] = thrust_controller_liftoff(t[k], X[k, :], P, az_filt_cache)
    Tmag = np.linalg.norm(Tcmd, axis=1)
    range_xy = np.linalg.norm(X[:, 0:2], axis=1)
    speed = np.linalg.norm(X[:, 3:6], axis=1)

    # 3D trajectory figure
    fig1 = plt.figure()
    ax3 = fig1.add_subplot(111, projection="3d")
    ax3.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=1.6)
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_title("Rocket Liftoff — 3-DoF")
    xr = max(200.0, np.max(np.abs(X[:, 0])) + 80.0)
    yr = max(200.0, np.max(np.abs(X[:, 1])) + 80.0)
    ax3.set_xlim(-xr, xr)
    ax3.set_ylim(-yr, yr)
    ax3.set_zlim(0.0, max(300.0, np.max(X[:, 2]) + 120.0))
    ax3.grid(True)

    # Log plots
    fig2, axs = plt.subplots(4, 1, figsize=(6, 9), sharex=True)
    epsy = 1e-6
    axs[0].plot(t, np.maximum(range_xy, epsy), linewidth=1.4)
    axs[0].set_yscale("log"); axs[0].grid(True); axs[0].set_ylabel("range_xy (m)")
    axs[0].set_title("Lateral Range (log)")

    axs[1].plot(t, np.maximum(speed, epsy), linewidth=1.4)
    axs[1].set_yscale("log"); axs[1].grid(True); axs[1].set_ylabel("speed (m/s)")
    axs[1].set_title("Speed (log)")

    axs[2].plot(t, np.maximum(Tmag, epsy), linewidth=1.4)
    axs[2].set_yscale("log"); axs[2].grid(True); axs[2].set_ylabel("|T| (N)")
    axs[2].set_title("Thrust (log)")

    axs[3].plot(t, np.maximum(X[:, 6], epsy), linewidth=1.4)
    axs[3].set_yscale("log"); axs[3].grid(True); axs[3].set_ylabel("mass (kg)")
    axs[3].set_xlabel("t (s)")
    axs[3].set_title("Mass (log)")

    xf = X[-1, :]
    print(
        f"End t={t[-1]:.2f} s | pos=({xf[0]:.2f},{xf[1]:.2f},{xf[2]:.2f}) m "
        f"| speed={np.linalg.norm(xf[3:6]):.2f} m/s | mass={xf[6]:.1f} kg"
    )

    plt.tight_layout()
    plt.show()


# ----------------- 3) Powered guided descent (PGD) demo -----------------

def dyn_pgd(_t: float, x: np.ndarray, P: Params) -> Tuple[np.ndarray, np.ndarray]:
    r = x[0:3]; v = x[3:6]; m = x[6]

    # Drag
    if P.CdA > 0 and vecnorm(v) > 1e-6:
        vhat = v / vecnorm(v)
        q = 0.5 * P.rho * (vecnorm(v) ** 2)
        aD = -(q * P.CdA / max(m, 1e-6)) * vhat
    else:
        aD = np.zeros(3)

    # Time-to-go estimate based on height and vertical speed
    h = max(r[2] - P.pad_r[2], 0.0)
    vz = v[2]
    tgo = max(P.tgo_min, 1.5 * h / max(abs(vz) + 1.0, 1.0))

    # Desired terminal velocity (slow down near ground)
    v_ref = np.array([0.0, 0.0, -min(P.v_touch, 0.5 + 0.1 * h)])

    # Errors
    e_r = (P.pad_r - r).copy()
    e_v = (v_ref - v)

    # Optional glide slope constraint (keep inside a cone)
    if P.gs_enable:
        rxy = r[0:2]; rng = vecnorm(rxy)
        if h > 0.0:
            max_rng = h * np.tan(P.gs_angle)
            if rng > max_rng:
                push_dir = rxy / max(rng, 1e-6)
                e_r[0:2] = e_r[0:2] - P.gs_k * (rng - max_rng) * push_dir

    # Desired acceleration (PD + gravity/drag compensation)
    a_des = (P.k_pos * e_r) / (tgo ** 2) + (P.k_vel * e_v) / tgo - P.g - aD

    # Raw thrust
    T_des = m * a_des

    # Gimbal limit relative to +z axis (cone angle)
    ez = np.array([0.0, 0.0, 1.0])
    Tn = vecnorm(T_des)
    if Tn > 1e-9:
        cosang = np.clip(np.dot(T_des, ez) / Tn, -1.0, 1.0)
        ang = np.arccos(cosang)
        if ang > P.theta_gim:
            axis_perp = T_des - np.dot(T_des, ez) * ez
            if vecnorm(axis_perp) < 1e-9:
                axis_perp = np.array([1.0, 0.0, 0.0])
            else:
                axis_perp = axis_perp / vecnorm(axis_perp)
            T_des = Tn * (np.cos(P.theta_gim) * ez + np.sin(P.theta_gim) * axis_perp)
    else:
        T_des = P.Tmin * ez

    # Throttle limits
    Tn = vecnorm(T_des)
    if Tn < P.Tmin:
        T_cmd = (P.Tmin / max(Tn, 1e-9)) * T_des
    elif Tn > P.Tmax:
        T_cmd = (P.Tmax / Tn) * T_des
    else:
        T_cmd = T_des

    rdot = v
    vdot = (1.0 / max(m, 1e-6)) * T_cmd + P.g + aD
    mdot = -vecnorm(T_cmd) / (P.Isp * P.g0)
    xdot = np.hstack([rdot, vdot, mdot])
    return xdot, T_cmd


def touchdown_event_pgd(_t: float, x: np.ndarray, _P: Params) -> float:
    """Terminate when z -> 0 (touchdown)."""
    return x[2]


def fuelout_event_pgd(_t: float, x: np.ndarray, _P: Params) -> float:
    """Terminate when mass <= 50 kg (near dry)."""
    return x[6] - 50.0


def run_pgd_demo():
    P = Params(
        g=np.array([0.0, 0.0, -9.80665]),
        rho=1.20,
        CdA=0.5 * 1.0,
        Isp=280.0,
        g0=9.80665,
        Tmin=0.10 * 1.5e5,
        Tmax=1.5e5,
    )
    P.theta_gim = np.deg2rad(15.0)
    P.v_touch = 0.75
    P.pad_r = np.array([0.0, 0.0, 0.0])
    P.gs_enable = True
    P.gs_angle = np.deg2rad(75.0)
    P.gs_k = 0.6
    P.k_pos = 0.9
    P.k_vel = 1.8
    P.tgo_min = 2.5

    r0 = np.array([250.0, -150.0, 600.0])
    v0 = np.array([-35.0, 15.0, -55.0])
    m0 = 2000.0
    x0 = np.hstack([r0, v0, m0])

    tspan = (0.0, 120.0)

    fun = lambda t, x: dyn_pgd(t, x, P)[0]
    evts = [
        lambda t, x: touchdown_event_pgd(t, x, P),
        lambda t, x: fuelout_event_pgd(t, x, P),
    ]
    for e in evts:
        # Bind event attributes as required by solve_ivp
        e.terminal = True
        e.direction = -1

    sol = solve_ivp(
        fun, tspan, x0, method="RK45",
        rtol=1e-7, atol=1e-8,
        events=evts, dense_output=False, max_step=0.02
    )
    t = sol.t
    X = sol.y.T

    # Recompute thrust
    Tcmd = np.zeros((len(t), 3))
    for i in range(len(t)):
        _, Ti = dyn_pgd(t[i], X[i, :], P)
        Tcmd[i, :] = Ti
    Tmag = np.linalg.norm(Tcmd, axis=1)

    # Plots
    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=1.6)
    ax3.scatter(0.0, 0.0, 0.0, marker='x')  # landing point
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_title("Powered Guided Descent: 3-DoF Trajectory")
    ax3.grid(True)
    ax3.set_box_aspect((1, 1, 1))

    fig2, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(t, X[:, 0:3], linewidth=1.4)
    axs[0].grid(True); axs[0].set_ylabel("r (m)")
    axs[0].legend(["x", "y", "z"], loc="best")
    axs[0].set_title("States")

    axs[1].plot(t, X[:, 3:6], linewidth=1.4)
    axs[1].grid(True); axs[1].set_ylabel("v (m/s)")
    axs[1].legend(["v_x", "v_y", "v_z"], loc="best")

    axs[2].plot(t, X[:, 6], linewidth=1.4)
    axs[2].grid(True); axs[2].set_ylabel("m (kg)")
    axs[2].set_xlabel("t (s)")

    fig3 = plt.figure()
    axT = fig3.add_subplot(111)
    axT.plot(t, Tmag, linewidth=1.6)
    axT.grid(True)
    axT.set_xlabel("t (s)")
    axT.set_ylabel("|T| (N)")
    axT.set_title("Thrust Magnitude")
    axT.axhline(P.Tmin, linestyle="--")
    axT.axhline(P.Tmax, linestyle="--")
    axT.legend(["|T|", "Tmin", "Tmax"])

    xf = X[-1, :]
    print(
        f"Touchdown t={t[-1]:.2f} s | pos=({xf[0]:.2f},{xf[1]:.2f},{xf[2]:.2f}) m "
        f"| speed={np.linalg.norm(xf[3:6]):.2f} m/s | mass={xf[6]:.1f} kg"
    )

    plt.tight_layout()
    plt.show()


# ----------------- 4) 3D linked liftoff animation with simple rocket mesh -----------------

def _make_cylinder(R=1.0, H=6.0, n=24, z0=0.0):
    """Return (V, F); V: Nx3 vertices; F: list of triangle faces (two per side quad)."""
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = R*np.cos(theta); y = R*np.sin(theta)
    V1 = np.c_[x, y, np.full_like(x, z0)]
    V2 = np.c_[x, y, np.full_like(x, z0+H)]
    V = np.vstack([V1, V2])
    F = []
    for i in range(n):
        j = (i+1) % n
        # Two triangles for the side quad
        F.append([i, j, n+j])
        F.append([i, n+j, n+i])
    return V, F


def _make_cone(R=1.0, H=3.0, n=24, z0=0.0):
    """Open cone (side triangles only)."""
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = R*np.cos(theta); y = R*np.sin(theta)
    Vbase = np.c_[x, y, np.full_like(x, z0)]
    Vtip  = np.array([[0.0, 0.0, z0+H]])
    V = np.vstack([Vbase, Vtip])
    tip = V.shape[0]-1
    F = [[i, (i+1)%n, tip] for i in range(n)]
    return V, F


def _make_fin(span=1.2, chord=2.5, thick=0.2, R_base=1.0, z0=0.0, angle=0.0):
    """Simple triangular fin extruded into a thin plate; returns mesh triangles."""
    p0 = np.array([R_base, 0.0, z0 + 0.05*chord])
    p1 = np.array([R_base, 0.0, z0 + 0.05*chord + chord])
    p2 = np.array([R_base+span, 0.0, z0 + 0.05*chord + 0.4*chord])
    Y  = np.array([+thick/2, -thick/2])
    Vtri = np.stack([p0,p1,p2], axis=0)
    V = np.vstack([
        np.c_[Vtri[:,0], np.full(3, Y[0]), Vtri[:,2]],
        np.c_[Vtri[:,0], np.full(3, Y[1]), Vtri[:,2]],
    ])
    # Rotate around z by angle
    ca, sa = np.cos(angle), np.sin(angle)
    Rz = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    V = (Rz @ V.T).T
    # Faces: two triangles per side + end caps
    F = []
    F += [[0,1,4],[0,4,3]]
    F += [[1,2,5],[1,5,4]]
    F += [[2,0,3],[2,3,5]]
    F += [[0,1,2],[3,5,4]]
    return V, F


def _apply_RT(V, R=None, t=None):
    """Rigid transform: V -> R*V + t"""
    W = V if R is None else (R @ V.T).T
    if t is not None:
        W = W + t
    return W


def _axis_align_R(z_hat=np.array([0,0,1.0])):
    """Return a rotation aligning body +Z to z_hat; identity if already aligned."""
    z = z_hat / (np.linalg.norm(z_hat) + 1e-12)
    tmp = np.array([1.0,0.0,0.0])
    if abs(np.dot(tmp,z)) > 0.95: tmp = np.array([0.0,1.0,0.0])
    x = np.cross(tmp, z); x = x / (np.linalg.norm(x)+1e-12)
    y = np.cross(z, x)
    R = np.stack([x,y,z], axis=1)  # columns are the new basis
    return R


def run_rocket_liftoff_anim():
    """
    Same dynamics as the stepviz version, but play it as a 3D animation.
    Keyboard:
      Space  pause/resume
      Up/Down increase/decrease playback speed
    """
    # Parameters
    P = Params(
        g=np.array([0.0, 0.0, -9.80665]),
        rho=1.20,
        CdA=0.5 * 1.0,
        Isp=280.0,
        g0=9.80665,
        Tmin=0.10 * 1.5e5,
        Tmax=1.5e5,
    )
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500.0])  # r,v,m
    tspan = (0.0, 60.0)

    # Integrate first, then animate
    az_cache = {"az": 0.0}
    fun = lambda t, x: dyn_with_controller_and_ground(t, x, P, az_cache)

    def evt_fuel(_t, x):
        return x[6] - 100.0
    evt_fuel.terminal = True
    evt_fuel.direction = -1

    sol = solve_ivp(fun, tspan, x0, method="RK45",
                    rtol=1e-7, atol=1e-8,
                    events=[evt_fuel], max_step=0.02)
    t = sol.t
    X = sol.y.T  # (N,7)

    stride = max(1, len(t)//1800)  # target ~1800 frames
    t = t[::stride]
    X = X[::stride, :]

    # Animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Rocket Liftoff — animation")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    # Ground square
    side = 200.0
    ax.plot([-side,-side, side, side, -side],
            [-side, side, side,-side, -side],
            [0,0,0,0,0], linewidth=1)

    xr = max(200.0, float(np.max(np.abs(X[:,0])) + 80.0))
    yr = max(200.0, float(np.max(np.abs(X[:,1])) + 80.0))
    zr = max(300.0, float(np.max(X[:,2]) + 120.0))
    ax.set_xlim(-xr, xr); ax.set_ylim(-yr, yr); ax.set_zlim(0.0, zr)
    ax.grid(True)

    (traj_line,) = ax.plot([], [], [], lw=1.5, color='tab:blue')
    (marker,)    = ax.plot([], [], [], 'o')

    info = ax.text2D(0.02, 0.92, "", transform=ax.transAxes)

    is_playing = True
    speed = 1  # advance this many indices per frame
    idx = 0

    def on_key(event):
        nonlocal is_playing, speed
        if event.key == ' ':
            is_playing = not is_playing
        elif event.key == 'up':
            speed = min(speed*2, 32)
        elif event.key == 'down':
            speed = max(1, speed//2)
    fig.canvas.mpl_connect('key_press_event', on_key)

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        marker.set_data([], [])
        marker.set_3d_properties([])
        info.set_text("")
        return traj_line, marker, info

    def update(_frame):
        nonlocal idx
        if is_playing:
            idx = min(idx + speed, len(t)-1)

        traj_line.set_data(X[:idx,0], X[:idx,1])
        traj_line.set_3d_properties(X[:idx,2])
        marker.set_data([X[idx,0]], [X[idx,1]])
        marker.set_3d_properties([X[idx,2]])

        info.set_text(f"t = {t[idx]:.2f} s\nz = {X[idx,2]:.1f} m\nspeed = {np.linalg.norm(X[idx,3:6]):.2f} m/s")
        return traj_line, marker, info

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=len(t), interval=20, blit=False)
    plt.show()


# ----------------- 5) Linked dashboard-style liftoff animation -----------------

def run_rocket_liftoff_anim_linked():
    # Parameters
    P = Params(
        g=np.array([0.0, 0.0, -9.80665]),
        rho=1.20,
        CdA=0.5 * 1.0,
        Isp=280.0,
        g0=9.80665,
        Tmin=0.10 * 1.5e5,
        Tmax=1.5e5,
    )
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500.0])
    tspan = (0.0, 60.0)

    # Integrate
    az_cache = {"az": 0.0}
    fun = lambda t, x: dyn_with_controller_and_ground(t, x, P, az_cache)

    def evt_fuel(_t, x):  # stop when m <= 100
        return x[6] - 100.0
    evt_fuel.terminal = True
    evt_fuel.direction = -1

    sol = solve_ivp(fun, tspan, x0, method="RK45",
                    rtol=1e-7, atol=1e-8, events=[evt_fuel], max_step=0.02)
    t = sol.t
    X = sol.y.T

    # Derived quantities
    Tcmd = np.zeros((len(t), 3))
    az_cache = {"az": 0.0}
    for k in range(len(t)):
        Tcmd[k, :] = thrust_controller_liftoff(t[k], X[k, :], P, az_cache)
    Tmag = np.linalg.norm(Tcmd, axis=1)
    range_xy = np.linalg.norm(X[:, :2], axis=1)
    speed = np.linalg.norm(X[:, 3:6], axis=1)
    mass = X[:, 6]

    # Frame control
    stride = max(1, len(t) // 1800)
    t, X = t[::stride], X[::stride]
    Tmag, range_xy, speed, mass = Tmag[::stride], range_xy[::stride], speed[::stride], mass[::stride]

    # Layout
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.4, 1.0], wspace=0.25, hspace=0.35)

    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axR = fig.add_subplot(gs[0, 1]); axV = fig.add_subplot(gs[1, 1])
    axT = fig.add_subplot(gs[2, 1]); axM = fig.add_subplot(gs[3, 1])

    ax3.set_title("Rocket Liftoff — animation (linked)")
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)"); ax3.set_zlabel("z (m)")
    ax3.view_init(elev=25, azim=35)
    side = 200.0
    ax3.plot([-side,-side, side, side, -side], [-side, side, side,-side, -side], [0,0,0,0,0], lw=1)
    xr = max(200.0, float(np.max(np.abs(X[:,0])) + 80.0))
    yr = max(200.0, float(np.max(np.abs(X[:,1])) + 80.0))
    zr = max(300.0, float(np.max(X[:,2]) + 120.0))
    ax3.set_xlim(-xr, xr); ax3.set_ylim(-yr, yr); ax3.set_zlim(0.0, zr)
    ax3.set_box_aspect((1, 1, 2))
    ax3.set_proj_type('persp')

    ax3.grid(True)
    (traj_line,) = ax3.plot([], [], [], lw=0.15, color='tab:blue')
    (marker,)    = ax3.plot([], [], [], 'o')

    # Right panels (base curves + moving points)
    epsy = 1e-9
    axR.plot(t, np.maximum(range_xy, epsy), lw=1.2, alpha=0.5); axR.set_yscale("log"); axR.grid(True)
    axR.set_ylabel("range_xy (m)"); axR.set_title("Lateral Range (log)")
    (mr,) = axR.plot([], [], 'o')

    axV.plot(t, np.maximum(speed, epsy), lw=1.2, alpha=0.5); axV.set_yscale("log"); axV.grid(True)
    axV.set_ylabel("speed (m/s)"); axV.set_title("Speed (log)")
    (mv,) = axV.plot([], [], 'o')

    axT.plot(t, np.maximum(Tmag, epsy), lw=1.2, alpha=0.5); axT.set_yscale("log"); axT.grid(True)
    axT.set_ylabel("|T| (N)"); axT.set_title("Thrust (log)")
    (mt,) = axT.plot([], [], 'o')

    axM.plot(t, np.maximum(mass, epsy), lw=1.2, alpha=0.5); axM.set_yscale("log"); axM.grid(True)
    axM.set_ylabel("mass (kg)"); axM.set_xlabel("t (s)"); axM.set_title("Mass (log)")
    (mm,) = axM.plot([], [], 'o')

    # HUD
    hud = ax3.text2D(0.02, 0.92, "", transform=ax3.transAxes, fontsize=11,
                     bbox=dict(facecolor="w", alpha=0.6, edgecolor="none"))

    # Interaction
    is_playing = True
    step = 1
    idx = 0
    nframes = len(t)

    def on_key(event):
        nonlocal is_playing, step
        if event.key == ' ':
            is_playing = not is_playing
        elif event.key == 'up':
            step = min(step * 2, 32)
        elif event.key == 'down':
            step = max(1, step // 2)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Build rocket mesh (auto-scaled to scene)
    scene_h = zr
    rocket_len = 0.5 * scene_h
    slender_ratio = 40.0

    L_body = 0.83 * rocket_len
    L_nose = 0.12 * rocket_len
    L_bell = 0.05 * rocket_len
    D_body = rocket_len / slender_ratio
    R_body = 0.5 * D_body
    n_side = 48

    V_body, F_body = _make_cylinder(R_body, L_body, n_side, z0=0.0)
    V_nose, F_nose = _make_cone(R_body*0.95, L_nose, n_side, z0=L_body)
    V_bell, F_bell = _make_cone(R_body*0.7,  L_bell, n_side, z0=-L_bell)

    # Four fins at 0, 90, 180, 270 deg
    fin_span, fin_chord, fin_thk = 1.4, 2.2, 0.12
    VF, FF = [], []
    for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
        Vfin, Ffin = _make_fin(fin_span, fin_chord, fin_thk, R_base=R_body, z0=0.2*L_body, angle=ang)
        VF.append(Vfin); FF.append(Ffin)
    V_fin = np.vstack(VF)
    F_fin = []
    base = 0
    for Vfin, Ffin in zip(VF, FF):
        F_fin += [[base+i for i in tri] for tri in Ffin]
        base += Vfin.shape[0]

    V0 = np.vstack([V_body, V_nose, V_bell, V_fin])
    F0 = F_body + [[len(V_body)+i for i in tri] for tri in F_nose] \
        + [[len(V_body)+len(V_nose)+i for i in tri] for tri in F_bell] \
        + [[len(V_body)+len(V_nose)+len(V_bell)+i for i in tri] for tri in F_fin]

    R_rocket = _axis_align_R(np.array([0,0,1.0]))
    t_rocket = X[0, :3]

    V_init = _apply_RT(V0, R_rocket, t_rocket)

    rocket_poly = Poly3DCollection([V_init[tri] for tri in F0],
                                   facecolor=(0.85,0.88,0.95,1.0),
                                   edgecolor=(0.2,0.2,0.25,0.3),
                                   linewidths=0.5)
    ax3.add_collection3d(rocket_poly)

    # Flame cone pointing -z (length will vary with thrust)
    flame_R0 = 0.6 * R_body
    flame_H0 = 25 * R_body
    V_flame0, F_flame0 = _make_cone(R=flame_R0, H=flame_H0, n=n_side, z0=-L_bell)
    flame_poly = Poly3DCollection(
        [V_flame0[tri] for tri in F_flame0],
        facecolor=(0.8, 0.0, 0.3, 0.6),
        edgecolor='none'
    )
    ax3.add_collection3d(flame_poly)

    def init():
        traj_line.set_data([], []); traj_line.set_3d_properties([])
        marker.set_data([], []); marker.set_3d_properties([])
        mr.set_data([], []); mv.set_data([], []); mt.set_data([], []); mm.set_data([], [])
        hud.set_text("")
        return traj_line, marker, mr, mv, mt, mm, hud

    def update(_frame):
        nonlocal idx
        if is_playing:
            idx = min(idx + step, nframes - 1)

        # Left 3D
        traj_line.set_data(X[:idx, 0], X[:idx, 1])
        traj_line.set_3d_properties(X[:idx, 2])
        marker.set_data([X[idx, 0]], [X[idx, 1]])
        marker.set_3d_properties([X[idx, 2]])

        # Orientation: keep vertical (use +Z). If you want velocity pointing, replace with v_hat.
        z_axis = np.array([0.0, 0.0, 1.0])
        R_rocket = _axis_align_R(z_axis)
        t_rocket = X[idx, :3]

        V_now = _apply_RT(V0, R_rocket, t_rocket)
        rocket_poly.set_verts([V_now[tri] for tri in F0])

        # Right panels moving markers
        mr.set_data([t[idx]], [max(range_xy[idx], epsy)])
        mv.set_data([t[idx]], [max(speed[idx], epsy)])
        mt.set_data([t[idx]], [max(Tmag[idx], epsy)])
        mm.set_data([t[idx]], [max(mass[idx], epsy)])

        # Flame intensity based on thrust vs hover
        hover_T = max(1e-6, X[idx, 6] * P.g0)
        thrust_frac = Tmag[idx] / hover_T
        alpha = float(np.clip((thrust_frac - 0.95) / 0.8, 0.0, 1.0))
        Hf = (1.0 + 5.0 * alpha) * flame_H0

        V_flame = V_flame0.copy()
        # Scale flame length along its local +z (tip direction), then flip towards -z below engine bell
        scale = Hf / max(flame_H0, 1e-9)
        V_flame[:, 2] = (V_flame[:, 2] - (-L_bell)) * scale + (-L_bell)

        Vf_now = _apply_RT(V_flame, R_rocket, t_rocket)
        flame_poly.set_verts([Vf_now[tri] for tri in F_flame0])
        fc = list(flame_poly.get_facecolor()[0]); fc[3] = 0.35 + 0.45 * alpha
        flame_poly.set_facecolor(tuple(fc))

        hud.set_text(
            f"{'▶' if is_playing else '⏸'} frame {idx+1}/{nframes}  step×{step}\n"
            f"t = {t[idx]:.2f} s   z = {X[idx,2]:.1f} m   |v| = {np.linalg.norm(X[idx,3:6]):.2f} m/s"
        )
        return traj_line, marker, mr, mv, mt, mm, hud

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=nframes, interval=20, blit=False)
    fig.ani = ani   # Keep a reference to prevent GC
    plt.show()


# ----------------- Script entry -----------------
if __name__ == "__main__":
    # Choose which demo to run:
    # run_rocket_liftoff_stepviz()
    # run_pgd_demo()
    # run_rocket_liftoff_anim()
    run_rocket_liftoff_anim_linked()
