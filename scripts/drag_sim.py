# rocket_freefall_teleop.py
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------- Utils ----------------
def vecnorm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))

def unit(v: np.ndarray) -> np.ndarray:
    n = vecnorm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)

# ---------------- Params & Aero ----------------
@dataclass
class Params:
    # environment & propulsion
    g: np.ndarray          # [m/s^2] gravity vector (downward)
    rho: float             # [kg/m^3] air density (constant here)
    Isp: float             # [s]
    g0: float              # [m/s^2]
    Tmin: float            # [N]
    Tmax: float            # [N]
    dry_mass: float        # [kg]

    # aerodynamic geometry (cylinder + cone)
    R_body: float = 0.5
    L_cyl:  float = 10.0
    L_cone: float = 3.0

    # aerodynamic coefficients
    C_Dn: float = 1.1
    C_Db: float = 0.15
    C_f:  float = 0.003

    # derived (filled by build_aero)
    A0: float = 0.0
    Sw: float = 0.0
    A_side: float = 0.0

def build_aero(P: Params) -> None:
    R, Lc, Ln = P.R_body, P.L_cyl, P.L_cone
    s_cone = np.hypot(R, Ln)
    P.A0 = float(np.pi * R**2)
    P.Sw = float(2.0 * np.pi * R * Lc + np.pi * R * s_cone)
    P.A_side = float(2.0 * R * Lc + R * Ln)

def drag_cyl_cone(vel: np.ndarray, P: Params, b_hat: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Return (Fd_vec, alpha, |D|, q)."""
    V2 = float(np.dot(vel, vel))
    if V2 < 1e-12:
        return np.zeros(3, dtype=float), 0.0, 0.0, 0.0
    V = np.sqrt(V2)
    u_v = vel / V
    q = 0.5 * P.rho * V2
    u_air = -u_v
    c = float(np.clip(np.dot(b_hat, u_air), -1.0, 1.0))  # cos(alpha)
    alpha = float(np.arccos(c))
    # projected area normal to velocity
    Aperp = P.A0 * abs(np.cos(alpha)) + P.A_side * np.sin(alpha)
    D = q * (P.C_Dn * Aperp + P.C_f * P.Sw * (np.cos(alpha) ** 2) + P.C_Db * P.A0 * abs(np.cos(alpha)))
    Fd = -D * u_v
    return Fd.astype(float), alpha, float(D), float(q)

# ---------------- Rocket meshes ----------------
def _make_cylinder(R=1.0, H=6.0, n=48, z0=0.0):
    th = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    x = R*np.cos(th); y = R*np.sin(th)
    V1 = np.c_[x, y, np.full_like(x, z0)]
    V2 = np.c_[x, y, np.full_like(x, z0+H)]
    V = np.vstack([V1, V2]).astype(float)
    F = []
    for i in range(n):
        j = (i+1) % n
        F += [[i, j, n+j], [i, n+j, n+i]]
    return V, F

def _make_cone(R=1.0, H=3.0, n=48, z0=0.0):
    th = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    x = R*np.cos(th); y = R*np.sin(th)
    Vbase = np.c_[x, y, np.full_like(x, z0)]
    Vtip  = np.array([[0.0, 0.0, z0+H]], dtype=float)
    V = np.vstack([Vbase, Vtip]).astype(float)
    tip = V.shape[0]-1
    F = [[i, (i+1) % n, tip] for i in range(n)]
    return V, F

def _make_cone_down(R=1.0, H=3.0, n=48, z0=0.0):
    """Cone with tip pointing downward (towards decreasing z)."""
    th = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    x = R*np.cos(th); y = R*np.sin(th)
    Vbase = np.c_[x, y, np.full_like(x, z0)]
    Vtip  = np.array([[0.0, 0.0, z0 - H]], dtype=float)
    V = np.vstack([Vbase, Vtip]).astype(float)
    tip = V.shape[0]-1
    F = [[i, (i+1) % n, tip] for i in range(n)]
    return V, F

def _make_fin(span=1.0, chord=2.0, thick=0.15, R_base=1.0, z0=0.0, angle=0.0):
    p0 = np.array([R_base, 0.0, z0 + 0.05*chord], dtype=float)
    p1 = np.array([R_base, 0.0, z0 + 0.05*chord + chord], dtype=float)
    p2 = np.array([R_base+span, 0.0, z0 + 0.05*chord + 0.4*chord], dtype=float)
    Y  = np.array([+thick/2, -thick/2], dtype=float)
    Vtri = np.stack([p0, p1, p2], axis=0)
    V = np.vstack([
        np.c_[Vtri[:,0], np.full(3, Y[0]), Vtri[:,2]],
        np.c_[Vtri[:,0], np.full(3, Y[1]), Vtri[:,2]],
    ]).astype(float)
    ca, sa = np.cos(angle), np.sin(angle)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    V = (Rz @ V.T).T
    F = []
    F += [[0,1,4],[0,4,3]]
    F += [[1,2,5],[1,5,4]]
    F += [[2,0,3],[2,3,5]]
    F += [[0,1,2],[3,5,4]]
    return V, F

def _apply_RT(V, R=None, t=None):
    W = V if R is None else (R @ V.T).T
    if t is not None:
        W = W + t
    return W

def _axis_align_R(z_hat=np.array([0.0,0.0,1.0], dtype=float)):
    z = z_hat / (np.linalg.norm(z_hat) + 1e-12)
    tmp = np.array([1.0,0.0,0.0], dtype=float)
    if abs(np.dot(tmp, z)) > 0.95:
        tmp = np.array([0.0,1.0,0.0], dtype=float)
    x = np.cross(tmp, z); x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

# ---------------- Dynamics (explicit gravity) ----------------
def dynamics(x: np.ndarray, P: Params, thrust_mag: float, b_hat_for_aero: np.ndarray):
    """Return (xdot, alpha, |D|, |Fg|, |T|)."""
    r = x[0:3]; v = x[3:6]; m = x[6]

    # Gravity force (explicit), drag, and thrust
    Fg = m * P.g                                 # gravity [N]
    Fd, alpha, Dmag, _ = drag_cyl_cone(v, P, b_hat_for_aero) if vecnorm(v) > 1e-6 else (np.zeros(3), 0.0, 0.0, 0.0)
    T  = np.array([0.0, 0.0, thrust_mag], dtype=float)   # teleop upward thrust [N]

    # Equations of motion
    rdot = v
    vdot = (Fg + Fd + T) / max(m, 1e-6)          # includes gravity explicitly
    mdot = -vecnorm(T) / (P.Isp * P.g0)

    xdot = np.hstack([rdot, vdot, mdot])
    return xdot, alpha, Dmag, vecnorm(Fg), vecnorm(T)

def rk4_step(x, f, dt):
    k1, *_ = f(x)
    k2, *_ = f(x + 0.5*dt*k1)
    k3, *_ = f(x + 0.5*dt*k2)
    k4, *_ = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------------- Interactive 3D animation ----------------
def run_freefall_teleop():
    # Start in free fall from 10,000 m
    m0 = 1000.0
    P = Params(
        g=np.array([0.0, 0.0, -9.80665], dtype=float),
        rho=1.20,
        Isp=280.0,
        g0=9.80665,
        Tmin=0.0,
        Tmax=3.5*m0*9.80665,    # ~3.5 g max thrust
        dry_mass=100.0,
        R_body=0.5, L_cyl=10.0, L_cone=3.0,
        C_Dn=1.1, C_Db=0.15, C_f=0.003,
    )
    build_aero(P)

    # State: [x y z vx vy vz m]^T
    x = np.array([0.0, 0.0, 10000.0,  0.0, 0.0, 0.0,  m0], dtype=float)

    # Rocket “nose” axis for aero (upright) and mesh
    body_up = np.array([0.0, 0.0, 1.0], dtype=float)

    # Teleop throttle -> thrust
    throttle = 0.0
    def thrust_from_throttle(thr):
        return float(np.clip(thr, 0.0, 1.0) * P.Tmax)

    # ---- Figure layout
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.45, 1.0], wspace=0.25, hspace=0.35)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axD = fig.add_subplot(gs[0, 1]); axA = fig.add_subplot(gs[1, 1])
    axZ = fig.add_subplot(gs[2, 1]); axV = fig.add_subplot(gs[3, 1])

    ax3.set_title("Free-fall from 10,000 m — Teleop Thrust (↑/↓) — Gravity + Cylinder+Cone Aero")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    ax3.view_init(elev=25, azim=35)
    side = 2000.0
    ax3.plot([-side,-side, side, side, -side], [-side, side, side,-side, -side], [0,0,0,0,0], lw=1, color='k')
    ax3.set_xlim(-500, 500); ax3.set_ylim(-500, 500); ax3.set_zlim(0.0, 10500.0)
    ax3.set_box_aspect((1, 1, 2.5))
    ax3.grid(True)

    (traj_line,) = ax3.plot([], [], [], lw=0.35, color='tab:blue')
    (marker,)    = ax3.plot([], [], [], 'o', color='tab:orange', markersize=4)
    hud = ax3.text2D(0.02, 0.92, "", transform=ax3.transAxes, fontsize=11,
                     bbox=dict(facecolor="w", alpha=0.65, edgecolor="none"))

    # Right-hand plots
    t_hist, D_hist, AoA_hist, Z_hist, V_hist = [], [], [], [], []
    (lD,) = axD.plot([], [], lw=1.2); axD.set_yscale("log"); axD.grid(True); axD.set_ylabel("Drag [N]"); axD.set_title("Drag (log)")
    (lA,) = axA.plot([], [], lw=1.2); axA.grid(True); axA.set_ylabel("AoA [deg]"); axA.set_title("Angle of Attack")
    (lZ,) = axZ.plot([], [], lw=1.2); axZ.grid(True); axZ.set_ylabel("Altitude [m]"); axZ.set_title("Altitude")
    (lV,) = axV.plot([], [], lw=1.2); axV.set_yscale("log"); axV.grid(True); axV.set_ylabel("Speed [m/s]"); axV.set_xlabel("t [s]"); axV.set_title("Speed (log)")

    # ---- Build rocket mesh (upright; nose +Z), engine bell, and downward flame
    scene_h = 10500.0
    rocket_len = 0.15 * scene_h
    slender_ratio = 18.0
    L_body = 0.80 * rocket_len
    L_nose = 0.15 * rocket_len
    L_bell = 0.05 * rocket_len
    D_body = rocket_len / slender_ratio
    R_vis  = 0.5 * D_body
    n_side = 64

    V_body, F_body = _make_cylinder(R_vis, L_body, n_side, z0=0.0)
    V_nose, F_nose = _make_cone(R_vis*0.98, L_nose, n_side, z0=L_body)
    V_bell, F_bell = _make_cone(R_vis*0.7,  L_bell, n_side, z0=-L_bell)   # engine bell below base
    flame_R0 = 0.8 * R_vis
    flame_H0 = 15 * R_vis
    V_flame0, F_flame0 = _make_cone_down(R=flame_R0, H=flame_H0, n=n_side, z0=-L_bell)

    # Fins
    fin_span, fin_chord, fin_thk = 1.2*R_vis, 0.25*L_body, 0.10*R_vis
    VF, FF = [], []
    for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
        Vfin, Ffin = _make_fin(fin_span, fin_chord, fin_thk, R_base=R_vis, z0=0.18*L_body, angle=ang)
        VF.append(Vfin); FF.append(Ffin)
    V_fin = np.vstack(VF)
    F_fin = []
    base = 0
    for Vfin, Ffin in zip(VF, FF):
        F_fin += [[base+i for i in tri] for tri in Ffin]
        base += Vfin.shape[0]

    V0 = np.vstack([V_body, V_nose, V_bell, V_fin])
    # Assemble faces with proper index offsets
    nb = len(V_body)
    nn = len(V_nose)
    nbell = len(V_bell)

    F0 = []
    F0 += F_body
    F0 += [[nb + idx for idx in tri] for tri in F_nose]
    F0 += [[nb + nn + idx for idx in tri] for tri in F_bell]
    F0 += [[nb + nn + nbell + idx for idx in tri] for tri in F_fin]


    R_rocket = _axis_align_R(np.array([0.0,0.0,1.0], dtype=float))
    def rocket_world(Vlocal, pos):
        return _apply_RT(Vlocal, R_rocket, pos)

    V_init = rocket_world(V0, x[0:3])
    Vf_init = rocket_world(V_flame0, x[0:3])

    rocket_poly = Poly3DCollection([V_init[tri] for tri in F0],
                                   facecolor=(0.85,0.88,0.95,1.0),
                                   edgecolor=(0.2,0.2,0.25,0.25),
                                   linewidths=0.4)
    flame_poly = Poly3DCollection([Vf_init[tri] for tri in F_flame0],
                                  facecolor=(0.95, 0.35, 0.05, 0.0),  # start hidden
                                  edgecolor='none')
    ax3.add_collection3d(rocket_poly)
    ax3.add_collection3d(flame_poly)

    # ---- Interaction
    is_playing = True
    dt = 0.02
    t = 0.0
    t_max = 600.0

    def on_key(event):
        nonlocal is_playing, throttle
        if event.key == ' ':
            is_playing = not is_playing
        elif event.key == 'up':
            throttle = min(1.0, throttle + 0.05)
        elif event.key == 'down':
            throttle = max(0.0, throttle - 0.05)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # ---- Animation callbacks
    xs, ys, zs = [], [], []

    def init():
        traj_line.set_data([], []); traj_line.set_3d_properties([])
        marker.set_data([], []); marker.set_3d_properties([])
        lD.set_data([], []); lA.set_data([], []); lZ.set_data([], []); lV.set_data([], [])
        hud.set_text("")
        return traj_line, marker, lD, lA, lZ, lV, rocket_poly, flame_poly, hud

    def step_sim():
        nonlocal x, t
        # terminate on ground with downward velocity ~0, or fuel out
        if x[2] <= 0.0 and x[5] <= 0.0:
            x[2] = 0.0; x[5] = 0.0
            return False

        if x[6] <= P.dry_mass:
            return False

        Tmag = thrust_from_throttle(throttle)
        b_hat = body_up  # upright body for AoA
        f = lambda xx: dynamics(xx, P, Tmag, b_hat)
        # RK4 (NOTE: f returns (xdot, alpha, D, |Fg|, |T|))
        x[:] = rk4_step(x, f, dt)
        # Recompute logs after stepping
        _, alpha, Dmag, Fgmag, Tmag_now = f(x)
        # prevent sinking
        if x[2] < 0.0:
            x[2] = 0.0
            if x[5] < 0.0: x[5] = 0.0
        t += dt
        return alpha, Dmag, Fgmag, Tmag_now

    def update(_frame):
        nonlocal t
        if is_playing and t < t_max:
            out = step_sim()
            if out is not False:
                alpha, Dmag, Fgmag, Tmag_now = out
                t_hist.append(t)
                D_hist.append(max(Dmag, 1e-9))
                AoA_hist.append(np.degrees(alpha))
                Z_hist.append(x[2])
                V_hist.append(vecnorm(x[3:6]))
                xs.append(x[0]); ys.append(x[1]); zs.append(x[2])

        # 3D updates
        traj_line.set_data(xs, ys); traj_line.set_3d_properties(zs)
        marker.set_data([x[0]], [x[1]]); marker.set_3d_properties([x[2]])

        V_now = rocket_world(V0, x[0:3])
        rocket_poly.set_verts([V_now[tri] for tri in F0])

        # Flame (downward) scaled by throttle
        flame_alpha = 0.0 if throttle < 1e-3 else (0.25 + 0.55*throttle)
        scale = (0.4 + 2.6*throttle)
        V_flame = V_flame0.copy()
        base_z = -L_bell
        V_flame[:, 2] = base_z - (np.abs(V_flame[:, 2] - base_z) * scale)
        Vf_now = rocket_world(V_flame, x[0:3])
        flame_poly.set_verts([Vf_now[tri] for tri in F_flame0])
        fc = list(flame_poly.get_facecolor()[0]); fc[3] = flame_alpha
        flame_poly.set_facecolor(tuple(fc))

        # Right-side plots
        lD.set_data(t_hist, D_hist)
        lA.set_data(t_hist, AoA_hist)
        lZ.set_data(t_hist, Z_hist)
        lV.set_data(t_hist, np.maximum(V_hist, 1e-9))
        for ax, y in [(axD, D_hist), (axA, AoA_hist), (axZ, Z_hist), (axV, V_hist)]:
            if len(y) > 5:
                ax.set_xlim(0.0, max(1.0, t_hist[-1]))
                ymin, ymax = float(np.min(y)), float(np.max(y))
                pad = 0.1*(ymax - ymin + 1e-6)
                # ensure positive range for log axes
                ymin = max(1e-9, ymin - pad) if ax.get_yscale() == 'log' else ymin - pad
                ax.set_ylim(ymin, ymax + pad)

        # HUD (shows gravity explicitly)
        hover_T = x[6]*P.g0
        hud.set_text(
            f"{'▶' if is_playing else '⏸'}  t={t:6.2f}s   z={x[2]:8.1f} m   |v|={vecnorm(x[3:6]):6.2f} m/s\n"
            f"throttle={throttle*100:5.1f}%   T={thrust_from_throttle(throttle):8.0f} N "
            f"(hover≈{hover_T:8.0f} N)   m={x[6]:7.1f} kg   g={vecnorm(P.g):.2f} m/s²"
        )
        return traj_line, marker, lD, lA, lZ, lV, rocket_poly, flame_poly, hud

    ani = animation.FuncAnimation(fig, update, init_func=init,
                              frames=200000, interval=16, blit=False)
    # Keep a reference so it isn't GC'd before show()
    fig.ani = ani

    plt.tight_layout()
    plt.show()

# -------------- Entry --------------
if __name__ == "__main__":
    run_freefall_teleop()
