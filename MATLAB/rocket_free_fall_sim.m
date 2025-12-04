function rocket_free_fall_sim()
    % ROCKET_FREE_FALL_SIM
    % Simulates a rocket in FREE FALL under Moon gravity.
    % Coordinate system: Polar (r, h) with fixed Yaw plane.
    % Visualizes the rocket as a 3D stick.
    
    clc; clear; close all;

    %% --- 1. USER CONFIGURATION ---
    user_r_init   = 500;    % Initial Horizontal Distance (meters)
    user_h_init   = 300;    % Initial Height (meters)
    user_yaw_init = 45;     % Initial Yaw (degrees) from Center Line
    user_speed    = 20;     % Initial Speed (m/s)
    
    % Simulation Settings
    t_span = [0 20];        % Max simulation duration (seconds)
    dt_vis = 0.05;          % Visualization update rate (faster for smoothness)

    %% --- 2. SYSTEM PARAMETERS ---
    p.g_moon = 1.62;        % Moon gravity m/s^2
    p.m_dry = 100;          % Dry mass (kg)
    p.m_fuel = 100;         % Fuel mass (kg)
    p.L_rocket = 10;        % Length of rocket "stick" for vis (m)
    
    %% --- 3. INITIAL CONDITIONS ---
    % Theta = -pi/2 means Horizontal, pointing towards target (-r direction)
    theta0 = -pi/2; 
    
    % Initial Velocity Decomposition
    dr0 = user_speed * sin(theta0); % Horizontal speed (negative -> towards target)
    dh0 = user_speed * cos(theta0); % Vertical speed
    
    X0 = [user_r_init;      % r
          dr0;              % dr
          user_h_init;      % h
          dh0;              % dh
          theta0;           % theta (pitch)
          0;                % dtheta
          p.m_dry + p.m_fuel]; % mass

    %% --- 4. SOLVE DYNAMICS ---
    options = odeset('RelTol',1e-4,'AbsTol',1e-4);
    [T_out, X_out] = ode45(@(t,x) rocket_dynamics(t, x, p), t_span, X0, options);

    %% --- 5. VISUALIZATION ---
    fig = figure('Color','k','Name','Rocket Free Fall 3D');
    
    % Enable Interactive Rotation immediately
    rotate3d on; 
    
    hold on; grid on; axis equal;
    
    % 5a. Setup Static Environment
    % Calculate plot limits to prevent auto-scaling jitter
    limit_range = user_r_init + 100;
    xlim([-limit_range/2, limit_range]);
    ylim([-limit_range/2, limit_range]);
    zlim([0, user_h_init + 50]);
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Height (m)');
    view(user_yaw_init + 45, 30); % Initial Camera Angle
    
    % Draw Moon Surface (Gray Patch)
    s = limit_range * 1.5;
    patch([-s s s -s], [-s -s s s], [0 0 0 0], ...
          [0.2 0.2 0.2], 'EdgeColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5);
    
    % Draw Target Landing Pad (3D Green Ring)
    theta_circ = linspace(0, 2*pi, 50);
    pad_r = 10; % Visual size of landing pad
    plot3(pad_r*cos(theta_circ), pad_r*sin(theta_circ), zeros(size(theta_circ)), ...
          'g-', 'LineWidth', 2);
    plot3(0, 0, 0, 'gx', 'MarkerSize', 15, 'LineWidth', 3);
    
    % 5b. Animation Objects
    % Rocket Body (Thick Red Line)
    rocket_line = plot3([0 0], [0 0], [0 0], 'r', 'LineWidth', 5);
    % Rocket Nose Marker (Yellow Dot to distinguish orientation)
    nose_marker = plot3(0,0,0, 'y.', 'MarkerSize', 15);
    % Trajectory Trail (Cyan Dotted)
    traj_line = plot3(0,0,0, 'c:', 'LineWidth', 1);
    
    % Info Text
    title_handle = title('Initializing...', 'Color', 'w', 'FontSize', 14);
    set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

    % Pre-calculate 3D coordinates based on fixed yaw
    yaw_rad = deg2rad(user_yaw_init);
    x_glob = X_out(:,1) * cos(yaw_rad);
    y_glob = X_out(:,1) * sin(yaw_rad);
    z_glob = X_out(:,3);
    
    % Pause briefly to let the user see the setup
    pause(1);

    % 5c. Animation Loop
    for i = 1:length(T_out)
        r = X_out(i,1);
        h = X_out(i,3);
        theta = X_out(i,5);
        
        % Check for crash
        if h <= 0
            h = 0; % Clamp to ground
            status = "IMPACT";
        else
            status = "FREE FALL";
        end
        
        % --- Calculate Rocket Stick Orientation ---
        % Center of Mass
        curr_pos = [x_glob(i), y_glob(i), h];
        
        % Vector components in the vertical plane (r, h)
        % theta = -90 (nose points to target/-r) -> vec_r = -1
        vec_r = sin(theta); 
        vec_h = cos(theta);
        
        % Project r-component into 3D (x, y)
        % This vector points from Tail towards Nose
        stick_vec = [vec_r * cos(yaw_rad), vec_r * sin(yaw_rad), vec_h];
        
        % Define endpoints (Stick centered on CoM)
        tip = curr_pos + stick_vec * (p.L_rocket / 2);
        tail = curr_pos - stick_vec * (p.L_rocket / 2);
        
        % Update Graphics
        set(rocket_line, 'XData', [tail(1) tip(1)], ...
                         'YData', [tail(2) tip(2)], ...
                         'ZData', [tail(3) tip(3)]);
                     
        set(nose_marker, 'XData', tip(1), 'YData', tip(2), 'ZData', tip(3));
                     
        set(traj_line, 'XData', x_glob(1:i), ...
                       'YData', y_glob(1:i), ...
                       'ZData', z_glob(1:i));
                   
        % Update Title
        t_str = sprintf('Time: %.2fs | Alt: %.1fm | Dist: %.1fm\nStatus: %s (Rotate View Enabled)', ...
            T_out(i), h, r, status);
        set(title_handle, 'String', t_str);
        
        % Force draw event (allows interaction)
        drawnow limitrate; 
        
        % Stop if crashed
        if h <= 0
            set(title_handle, 'String', [t_str ' - SIMULATION END']);
            break;
        end
        
        % Control playback speed
        % Interpolating delay based on simulation time steps
        if i < length(T_out)
            real_dt = T_out(i+1) - T_out(i);
            pause(real_dt); 
        end
    end
end

%% --- DYNAMICS FUNCTION ---
function dX = rocket_dynamics(~, X, p)
    % Unpack State
    % r = X(1); dr = X(2);
    % h = X(3); dh = X(4);
    % theta = X(5); dtheta = X(6);
    % m = X(7);
    
    % Free Fall Dynamics (No Thrust, No Torque)
    dr_dt = X(2);
    ddr = 0;
    dh_dt = X(4);
    ddh = -p.g_moon;
    dtheta_dt = X(6);
    ddtheta = 0;
    dm = 0;

    dX = [dr_dt; ddr; dh_dt; ddh; dtheta_dt; ddtheta; dm];
end