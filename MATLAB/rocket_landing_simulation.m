function rocket_landing_simulation()
    % ROCKET_LANDING_SIMULATION
    % Simulates a variable mass rocket in Moon gravity using Lagrangian dynamics.
    % Coordinate system: Polar (r, h) with fixed Yaw plane.
    % Visualizes the rocket as a stick.
    
    clc; clear; close all;

    %% --- 1. USER CONFIGURATION ---
    % Change these values as requested
    user_r_init   = 500;    % Initial Horizontal Distance (meters)
    user_h_init   = 300;    % Initial Height (meters)
    user_yaw_init = 45;     % Initial Yaw (degrees) from Center Line
    user_speed    = 20;     % Initial Speed (m/s)
    
    % Simulation Settings
    t_span = [0 20];        % Simulation duration (seconds)
    dt_vis = 0.1;           % Visualization update rate

    %% --- 2. SYSTEM PARAMETERS ---
    p.g_moon = 1.62;        % Moon gravity m/s^2 [cite: 9]
    p.g_earth = 9.81;       % Standard gravity for Isp calc [cite: 108]
    p.m_dry = 100;          % Dry mass (kg)
    p.m_fuel = 100;         % Fuel mass (kg)
    p.Isp = 300;            % Specific Impulse (s)
    p.L_rocket = 5;         % Length of rocket "stick" for vis (m)
    p.I = (1/12)*200*(5^2); % Approx Moment of Inertia (kg*m^2)
    
    % Thrust Constraints
    % Max Thrust = 5 * Body Weight (based on fully fueled mass)
    % Weight = (100+100) * 1.62. Max Thrust = 5 * Weight.
    p.max_thrust = 5 * (p.m_dry + p.m_fuel) * p.g_moon; 
    
    %% --- 3. INITIAL CONDITIONS ---
    % State Vector: [r, dr, h, dh, theta, dtheta, mass]
    % Note: Theta is 0 for Vertical (Up), pi/2 for Horizontal
    
    % Initial Orientation: Horizontal pointing towards target
    % "Head" points to target, so engine points away.
    theta0 = -pi/2; % -90 degrees (Horizontal, nose towards -r)
    
    % Initial Velocity: Aligned with rocket head direction
    dr0 = user_speed * sin(theta0); % Horizontal speed
    dh0 = user_speed * cos(theta0); % Vertical speed
    
    X0 = [user_r_init;      % r
          dr0;              % dr
          user_h_init;      % h
          dh0;              % dh
          theta0;           % theta (pitch)
          0;                % dtheta (angular velocity)
          p.m_dry + p.m_fuel]; % mass

    %% --- 4. SOLVE DYNAMICS ---
    % Using ode45 for integration
    options = odeset('RelTol',1e-4,'AbsTol',1e-4);
    [T_out, X_out] = ode45(@(t,x) rocket_dynamics(t, x, p), t_span, X0, options);

    %% --- 5. VISUALIZATION ---
    figure('Color','k','Name','Rocket Landing Simulation');
    hold on; grid on; axis equal;
    
    % Setup Scene
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Height (m)');
    view(user_yaw_init + 45, 30); % Adjust camera based on yaw
    
    % Draw Moon Surface
    patch([-1000 1000 1000 -1000], [-1000 -1000 1000 1000], [0 0 0 0], ...
          [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    
    % Draw Target
    plot3(0, 0, 0, 'rx', 'MarkerSize', 15, 'LineWidth', 3);
    viscircles([0 0], 0.5, 'Color', 'g'); % 0.5m target threshold 
    
    % Animation Loop
    rocket_line = plot3([0 0], [0 0], [0 0], 'w', 'LineWidth', 4); % The "Stick"
    traj_line = plot3(0,0,0, 'c:', 'LineWidth', 1);
    title_handle = title('');
    
    % Pre-calculate 3D coordinates based on fixed yaw
    % Convert polar r to x,y
    yaw_rad = deg2rad(user_yaw_init);
    x_pos = X_out(:,1) * cos(yaw_rad);
    y_pos = X_out(:,1) * sin(yaw_rad);
    z_pos = X_out(:,3);
    
    for i = 1:length(T_out)
        r = X_out(i,1);
        h = X_out(i,3);
        theta = X_out(i,5);
        
        % Current 3D Position
        curr_pos = [x_pos(i), y_pos(i), z_pos(i)];
        
        % Calculate Rocket "Stick" endpoints for visualization
        % Orientation vector in the fixed vertical plane
        % Theta = 0 is UP (Z+). Theta = -90 is pointing towards target (-r)
        
        % Local components in the r-z plane
        vec_r = sin(theta); % Horizontal component
        vec_h = cos(theta); % Vertical component
        
        % Project r-component into 3D (x,y)
        stick_vec = [vec_r * cos(yaw_rad), vec_r * sin(yaw_rad), vec_h];
        
        % Define stick ends (Center of Mass in middle)
        tip = curr_pos + stick_vec * (p.L_rocket / 2);
        tail = curr_pos - stick_vec * (p.L_rocket / 2);
        
        % Update Graphics
        set(rocket_line, 'XData', [tail(1) tip(1)], ...
                         'YData', [tail(2) tip(2)], ...
                         'ZData', [tail(3) tip(3)]);
                     
        set(traj_line, 'XData', x_pos(1:i), ...
                       'YData', y_pos(1:i), ...
                       'ZData', z_pos(1:i));
                   
        % Dynamic Title
        status = "Flying";
        if h <= 0; status = "CRASH/LANDED"; end
        
        t_str = sprintf('Time: %.2fs | Alt: %.1fm | Dist: %.1fm | Fuel: %.1fkg\nStatus: %s', ...
            T_out(i), h, r, X_out(i,7)-p.m_dry, status);
        set(title_handle, 'String', t_str, 'Color', 'w');
        set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
        
        % Target Check (Simple)
        if r < 0.5 && h < 0.1
             text(0,0,5, 'TARGET REACHED', 'Color', 'g', 'FontSize', 20);
        end

        drawnow;
        
        % Stop if hit ground
        if h <= 0
            break;
        end
        
        % Time step delay
        if i < length(T_out)
            pause((T_out(i+1) - T_out(i)));
        end
    end
end

%% --- DYNAMICS FUNCTION ---
function dX = rocket_dynamics(t, X, p)
    % Unpack State
    r = X(1); dr = X(2);
    h = X(3); dh = X(4);
    theta = X(5); dtheta = X(6);
    m = X(7);
    
    % Clamp mass to dry mass
    if m < p.m_dry
        m = p.m_dry;
    end
    
    % --- CONTROL LOGIC (Simple Open Loop) ---
    % Note: User asked for dynamics, but without control it just falls.
    % We apply a simple constant retro-thrust to show dynamics.
    
    % Determine Thrust (Simple logic: if fuel left, burn at 80%)
    thrust_cmd = 0;
    if m > p.m_dry
        thrust_cmd = 0.8 * p.max_thrust; % 80% Throttle
    end
    
    % Determine Torque (Simple logic: try to rotate to upright 0 rad)
    % This acts as a damped spring to stabilize the stick
    kp = 100; kd = 200;
    desired_angle = 0; % Vertical
    torque_cmd = -kp*(theta - desired_angle) - kd*dtheta;
    
    % --- LAGRANGE DYNAMICS IMPLEMENTATION ---
    % Thrust Vector Decomposition
    % The thrust acts along the body axis. 
    % If theta=0 (Up), Thrust pushes Up (+h).
    % If theta=-pi/2 (Horizontal towards target), Thrust pushes Left (-r).
    
    F_rad = thrust_cmd * sin(theta); 
    F_vert = thrust_cmd * cos(theta);
    
    % 1. Radial Acceleration (Lagrange d/dt(dL/dr_dot) - dL/dr = F_r)
    % m*r_ddot = F_rad
    ddr = F_rad / m;
    
    % 2. Vertical Acceleration
    % m*h_ddot + m*g = F_vert
    ddh = (F_vert / m) - p.g_moon;
    
    % 3. Angular Acceleration
    % I*theta_ddot = Torque
    ddtheta = torque_cmd / p.I;
    
    % 4. Mass Depletion 
    dm = -thrust_cmd / (p.Isp * p.g_earth);
    if m <= p.m_dry
        dm = 0;
    end

    % Pack Derivative
    dX = [dr; ddr; dh; ddh; dtheta; ddtheta; dm];
end