%% Rocket Landing Simulation (6-DOF: Pitch, Yaw, Roll)
%  - Initial State: Horizontal Flight (AoA=0), Non-zero Roll
%  - Dynamics: 6-DOF Rigid Body with Gimbal Control
%  - Visuals: 3D Interactive + Pitch/Roll/Mass Plots
clear; close all; clc;

%% --- 1. CONFIGURATION ---
sim.dt = 0.05;          % Time step
sim.t_max = 25;         % Max duration
sim.g = 1.62;           % Moon gravity [m/s^2]

% Rocket Parameters
p.m_dry = 100;          % Dry Mass [kg]
p.m_fuel = 100;         % Fuel Mass [kg]
p.Isp = 300;            % Specific Impulse [s]
p.L = 8;                % Rocket Length [m]
p.r_cyl = 0.5;          % Rocket Radius [m]
p.max_thrust = 8000;    % Max Thrust [N]

% Inertia (Cylinder approx)
p.get_I_long = @(m) (1/12)*m*p.L^2;  % Pitch/Yaw Inertia
p.get_I_roll = @(m) 0.5*m*p.r_cyl^2; % Roll Inertia
p.arm = p.L/2;          % Gimbal lever arm

%% --- 2. INITIAL CONDITIONS ---
% 1. Position
x0 = 0; y0 = 0; z0 = 400;

% 2. Velocity (Horizontal Flight)
speed = 40; 
heading = deg2rad(30); % Flying 30 deg from X-axis
vx0 = speed * cos(heading);
vy0 = speed * sin(heading);
vz0 = 0; % Horizontal

% 3. Orientation (Angle of Attack = 0)
% Rocket body must align with Velocity vector.
% Convention: Pitch=0 is Horizontal. Pitch=90 is Vertical Up.
theta0 = 0;          % Pitch (Horizontal)
psi0   = heading;    % Yaw (Aligned with heading)
phi0   = deg2rad(45);% Initial ROLL angle (User Request)

% State: [x, vx, y, vy, z, vz, theta, dtheta, psi, dpsi, phi, dphi, mass]
X0 = [x0; vx0; y0; vy0; z0; vz0; ...
      theta0; 0; ...
      psi0; 0; ...
      phi0; 0; ...
      p.m_dry + p.m_fuel];

%% --- 3. RUN SIMULATION ---
options = odeset('RelTol', 1e-5, 'AbsTol', 1e-5);
[t, X] = ode45(@(t,x) dynamics(t, x, p), [0 sim.t_max], X0, options);

%% --- 4. VISUALIZATION ---
fig = figure('Color', 'w', 'Name', '6-DOF Rocket Landing', 'Position', [50 50 1400 800]);
rotate3d on; % INTERACTIVE 3D

% --- 3D VIEW (Main) ---
ax3d = subplot(3, 3, [1 4 7]);
hold(ax3d, 'on'); grid(ax3d, 'on'); axis(ax3d, 'equal');
xlabel(ax3d, 'X'); ylabel(ax3d, 'Y'); zlabel(ax3d, 'Alt');
title(ax3d, '3D Trajectory (Green Strip = Roll)');
view(ax3d, 45, 20);

% Ground & Target
patch(ax3d, [-200 800 800 -200], [-200 -200 800 800], [0 0 0 0], [0.9 0.9 0.9], 'EdgeAlpha', 0);
plot3(ax3d, 400, 400, 0, 'gx', 'MarkerSize', 15, 'LineWidth', 3); % Arbitrary Target
viscircles(ax3d, [400 400], 20, 'Color', 'g');

% Graphics Objects
rocket_body = plot3(ax3d, [0 0], [0 0], [0 0], 'k-', 'LineWidth', 4);
rocket_nose = plot3(ax3d, 0, 0, 0, 'r^', 'MarkerSize', 8, 'MarkerFaceColor','r');
roll_strip  = plot3(ax3d, [0 0], [0 0], [0 0], 'g-', 'LineWidth', 2); % To visualize Roll
thrust_vec  = plot3(ax3d, [0 0], [0 0], [0 0], 'm-', 'LineWidth', 1.5);
trail       = plot3(ax3d, X(:,1), X(:,3), X(:,5), 'b:', 'LineWidth', 1);

% --- DATA PLOTS ---
% Altitude
ax_h = subplot(3,3,2); title(ax_h, 'Altitude [m]'); grid on; hold on;
plot(ax_h, t, X(:,5), 'b', 'LineWidth', 1.5);

% Pitch Angle
ax_p = subplot(3,3,3); title(ax_p, 'Pitch (Theta) [deg]'); grid on; hold on;
plot(ax_p, t, rad2deg(X(:,7)), 'r', 'LineWidth', 1.5);
yline(ax_p, 90, 'k--', 'Vertical');

% Mass
ax_m = subplot(3,3,5); title(ax_m, 'Mass [kg]'); grid on; hold on;
plot(ax_m, t, X(:,13), 'k', 'LineWidth', 1.5);

% Roll Angle
ax_r = subplot(3,3,6); title(ax_r, 'Roll (Phi) [deg]'); grid on; hold on;
plot(ax_r, t, rad2deg(X(:,11)), 'g', 'LineWidth', 1.5);

% Controls
ax_c = subplot(3,3,[8 9]); title(ax_c, 'Thrust Command [N]'); grid on; hold on;
ctrl_line = plot(ax_c, NaN, NaN, 'm', 'LineWidth', 1.5);

% --- ANIMATION LOOP ---
% Pre-calculate controls for plotting
U_log = zeros(length(t), 1);
for k=1:length(t), u=flight_computer(t(k), X(k,:)', p); U_log(k)=u(1); end
set(ctrl_line, 'XData', t, 'YData', U_log);

fprintf('Click the plot to rotate view during animation.\n');
for i = 1:2:length(t)
    if ~ishandle(fig), break; end
    
    % Unpack State
    x_c = X(i,1); y_c = X(i,3); z_c = X(i,5);
    th = X(i,7); psi = X(i,9); phi = X(i,11);
    
    if z_c <= 0, title(ax3d, 'TOUCHDOWN'); break; end
    
    % --- COORDINATE TRANSFORMS (Body -> World) ---
    % Rotation Matrix R = Rz(psi) * Ry(theta) * Rx(phi)
    % Note: Theta=0 is Horizontal. Theta=90 is Vertical.
    % To match standard aerospace (Body X is axis), we treat Body X as Rocket Axis.
    
    % Basis Vectors in Body Frame
    b_axis = [1; 0; 0];  % Rocket Length Axis
    b_fin  = [0; 1; 0];  % Fin Axis (for Roll viz)
    
    % Euler Angles (Yaw=psi, Pitch=theta, Roll=phi)
    % Transform Body X (1,0,0) to World
    cp = cos(th); sp = sin(th);
    cy = cos(psi); sy = sin(psi);
    cr = cos(phi); sr = sin(phi);
    
    % Direction of Nose (World Frame)
    % Pure Geometry: Pitch elevates from XY plane. Yaw rotates in XY.
    dir_nose = [cos(th)*cos(psi); cos(th)*sin(psi); sin(th)];
    
    % Direction of Fin (Perpendicular to Nose, rotates with Roll)
    % This requires the full rotation matrix column 2
    % R[0,1] = cos(psi)sin(th)sin(phi) - sin(psi)cos(phi)
    % R[1,1] = sin(psi)sin(th)sin(phi) + cos(psi)cos(phi)
    % R[2,1] = -cos(th)sin(phi) -- wait, depends on Euler sequence.
    % Let's use standard Z-Y-X rotation matrix for visualization vector:
    dir_fin = [
        cos(psi)*sin(th)*sin(phi) - sin(psi)*cos(phi);
        sin(psi)*sin(th)*sin(phi) + cos(psi)*cos(phi);
        -cos(th)*sin(phi)
    ];
    
    % Scale
    nose_pt = [x_c; y_c; z_c] + dir_nose * (p.L/2);
    tail_pt = [x_c; y_c; z_c] - dir_nose * (p.L/2);
    fin_pt  = [x_c; y_c; z_c] + dir_fin * (p.L/3); % Stick sticking out side
    
    % Draw
    set(rocket_body, 'XData', [tail_pt(1) nose_pt(1)], ...
                     'YData', [tail_pt(2) nose_pt(2)], ...
                     'ZData', [tail_pt(3) nose_pt(3)]);
    
    set(rocket_nose, 'XData', nose_pt(1), 'YData', nose_pt(2), 'ZData', nose_pt(3));
    
    set(roll_strip,  'XData', [x_c fin_pt(1)], ...
                     'YData', [y_c fin_pt(2)], ...
                     'ZData', [z_c fin_pt(3)]);
                 
    % Thrust Plume
    T_mag = U_log(i);
    plume_len = (T_mag/p.max_thrust) * 5;
    plume_pt = tail_pt - dir_nose * plume_len;
    
    set(thrust_vec,  'XData', [tail_pt(1) plume_pt(1)], ...
                     'YData', [tail_pt(2) plume_pt(2)], ...
                     'ZData', [tail_pt(3) plume_pt(3)]);
                 
    % Follow Camera (Optional - keep centered on rocket)
    % xlim(ax3d, [x_c-100 x_c+100]); ylim(ax3d, [y_c-100 y_c+100]);
    
    drawnow limitrate;
    pause(0.01);
end

%% --- 5. DYNAMICS ---
function dX = dynamics(t, state, p)
    % Unpack: x, vx, y, vy, z, vz, th, dth, psi, dpsi, phi, dphi, m
    vx = state(2); vy = state(4); vz = state(6);
    th = state(7); dth = state(8);
    psi = state(9); dpsi = state(10);
    phi = state(11); dphi = state(12);
    m = state(13);
    
    % Control
    u = flight_computer(t, state, p);
    T = u(1);
    d_pitch = u(2); % Gimbal Pitch
    d_yaw   = u(3); % Gimbal Yaw
    % Roll control assumed via small RCS torque
    tau_roll_cmd = u(4); 
    
    % --- FORCES (World Frame) ---
    % Thrust is aligned with Body Axis + Gimbal Deflection
    % Effective Thrust Angle: Theta_eff = Theta + d_pitch
    %                         Psi_eff   = Psi + d_yaw
    
    eff_th = th + d_pitch;
    eff_psi = psi + d_yaw;
    
    Fx = T * cos(eff_th) * cos(eff_psi);
    Fy = T * cos(eff_th) * sin(eff_psi);
    Fz = T * sin(eff_th);
    
    ax = Fx / m;
    ay = Fy / m;
    az = (Fz / m) - 1.62;
    
    % --- TORQUES (Body Frame) ---
    % Gimbal creates torque perp to body axis
    tau_pitch = -T * sin(d_pitch) * p.arm;
    tau_yaw   = -T * sin(d_yaw) * p.arm;
    tau_roll  = tau_roll_cmd; % Direct torque input
    
    % --- INERTIA & ANGULAR ACCEL ---
    I_long = p.get_I_long(m);
    I_roll = p.get_I_roll(m);
    
    % Euler's Eq approx (Decoupled for stability in simple sim)
    ddth = tau_pitch / I_long;
    ddpsi = tau_yaw / I_long;
    ddphi = tau_roll / I_roll;
    
    % Mass
    dm = -T / (p.Isp * 9.81);
    if m <= p.m_dry, dm=0; T=0; end
    
    dX = [vx; ax; vy; ay; vz; az; dth; ddth; dpsi; ddpsi; dphi; ddphi; dm];
end

%% --- 6. FLIGHT COMPUTER ---
function u = flight_computer(t, state, p)
    % Controller: Brake Horizontal -> Orient Vertical -> Soft Land
    z = state(5); vz = state(6);
    th = state(7); dth = state(8);
    phi = state(11); dphi = state(12);
    
    % 1. ROLL STABILIZATION
    % Try to zero out the roll
    target_phi = 0;
    tau_roll = 100 * (target_phi - phi) - 50 * dphi;
    
    % 2. PITCH LOGIC
    if z > 50
        % "Retrograde": Pitch Up to 45-60 deg to kill horizontal speed
        target_theta = deg2rad(45); 
    else
        % "Landing": Vertical (90 deg)
        target_theta = deg2rad(90);
    end
    
    % PD Pitch Gimbal
    target_theta = clamp(target_theta, deg2rad(-10), deg2rad(100));
    cmd_pitch = 1.5 * (target_theta - th) - 2.0 * dth;
    
    % 3. YAW LOGIC (Hold Heading)
    cmd_yaw = 0; % Simplified for this demo
    
    % 4. THROTTLE LOGIC
    if z > 100
        T_cmd = p.max_thrust * 0.8; % Burn
    else
        % Hover / Descent
        v_target = -2.0; % Descent at 2 m/s
        err_v = v_target - vz;
        T_hover = state(13) * 1.62 / sin(th+0.01); % Compensate gravity
        T_cmd = T_hover + 2000 * err_v;
    end
    
    % Limits
    u = [clamp(T_cmd, 0, p.max_thrust), ...
         clamp(cmd_pitch, -0.2, 0.2), ...
         clamp(cmd_yaw, -0.2, 0.2), ...
         clamp(tau_roll, -100, 100)];
end

function y = clamp(x, lo, hi)
    y = max(lo, min(x, hi));
end