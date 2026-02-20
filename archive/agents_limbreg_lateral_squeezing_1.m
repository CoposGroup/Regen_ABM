% Agent-based stochastic model
% Cells represented as point particles
% Brownian motion and repulsion
%
% CC (Jan 2024)

close all; % Close all figures
% clc; % Clear the command window
clear; % Clear workspace variables

load('cellinitialization_n100','pos0','Ncells'); % Load initial cell positions and number of cells
N = Ncells; % Number of cells
n = length(pos0); % Number of initial positions

video_flag = 0; % Flag to create video
if(video_flag)
    vidObj = VideoWriter('lateral_squeezing_kb5.mp4','MPEG-4'); % Create video object
    open(vidObj); % Open the video object
end

% Parameters
dt = 0.00001; % Time step
Tmax = 5.0 * 2; % Maximum time
mu = 1; % Unused parameter
dl_crit = 0.1; % Critical distance for computing repulsive force
xi = 1.0; % Drag coefficient
kb = 5.0; % Stiffness of epidermis
kcoll = 0.08; % Epidermis-cell collision constant
kdiv = 0.1/2; % Proliferation rate
offset = 0.1; % Distance between mother and daughter cell

pos = pos0; % Initial positions
pre_pos = pos; % Previous positions (unused)

% A large array for positions
cells_max = 5 * N; % Maximum number of cells
pos = NaN(cells_max,2); % Position array
v = NaN(cells_max,2); % Velocity array
tau = NaN(cells_max,1); % Unused array
v(1:N,:) = zeros(N,2); % Initialize velocities to zero
tau(1:N,:) = zeros(N,1); % Initialize tau to zero
pos(1:N, :) = pos0; % Set initial positions

% Division status
division_status = false(cells_max, 1); % Initialize division status

% Division interval
div_interval = 0.4; % Interval between cell divisions
next_div_time = div_interval; % Next division time

% Define semi-circle
semi_circle_center = [0, 0]; % Center of the semi-circle
semi_circle_radius = 1.5; % Radius of the semi-circle
theta = linspace(3*pi/2,5*pi/2); % Angles for the semi-circle
xb_semi_circle = semi_circle_radius * cos(theta) + semi_circle_center(1); % X-coordinates of the semi-circle
yb_semi_circle = semi_circle_radius * sin(theta) + semi_circle_center(2); % Y-coordinates of the semi-circle

% Define the vertical line segments
x_vert = [1.2, 1.2]; % X-coordinates of the vertical line

diffs = abs(xb_semi_circle - x_vert(1)); % Differences between semi-circle and vertical line
[~, sorted_indices] = sort(diffs); % Sort the differences
closest_indices = sorted_indices(1:2); % Indices of the two closest points
val_yb = yb_semi_circle(closest_indices); % Y-coordinates of the closest points
val_xb = xb_semi_circle(closest_indices); % X-coordinates of the closest points

y_vert = [val_yb(1),val_yb(2)]; % Y-coordinates for the vertical line

% Find mean spacing
xb_space = xb_semi_circle(1:min(closest_indices)); % X-coordinates for spacing
yb_space = yb_semi_circle(1:min(closest_indices)); % Y-coordinates for spacing
for i=1:length(xb_space)-1
    ds = sqrt((xb_space(i+1)-xb_space(i))^2+(yb_space(i+1)-yb_space(i))^2); % Distance between adjacent points
end
avg_ds = mean(ds); % Average distance

y_v = y_vert(2):avg_ds:y_vert(1); % Y-coordinates for vertical points
x_v = 1.2*ones(1,length(y_v)); % X-coordinates for vertical points

% Combine the semi-circle and the vertical line segment in correct order
xb = [xb_semi_circle(1:min(closest_indices)),x_v,xb_semi_circle(max(closest_indices):end)];
yb = [yb_semi_circle(1:min(closest_indices)),y_v,yb_semi_circle(max(closest_indices):end)];

Xb0 = [xb;yb]'; % Initial boundary points
Xb = [xb;yb]'; % Current boundary points
dsb = sqrt( (Xb0(1,1)-Xb0(2,1))^2 + (Xb0(1,2)-Xb0(2,2))^2 ); % Initial distance between the first two points

% Make a difference matrix for the boundary points
Nb = length(Xb); % Number of boundary points
e = ones(Nb,1); % Vector of ones
Db = spdiags([-e e],[0 1],Nb,Nb); % Difference matrix
Db(Nb,1) = 1; % Wrap around

% Tether points/top channel resting length
blp0 = Db * Xb0; % Rest length of edges
blp0 = sqrt(blp0(:,1).^2 + blp0(:,2).^2); % Distance between adjacent points
blm0 = Db' * Xb0; % Rest length of edges
blm0 = sqrt(blm0(:,1).^2 + blm0(:,2).^2); % Distance between adjacent points

% Alive status
alive = NaN(length(pos),1); % Alive status array
not_NaN_rows = ~any(isnan(pos),2); % Rows that are not NaN
alive(not_NaN_rows & isnan(alive)) = 1; % Set alive status to 1

% Number of daughter cells
n_daughter = 0;

upper_bdry = 1; % Upper boundary
lower_bdry = -1; % Lower boundary

% Identify cells above y=1
cells_above_y1 = find(pos(:,2) >= upper_bdry); % Cells above y=1
num_cells_above_y1 = length(cells_above_y1); % Number of cells above y=1

% Randomly select 50% of the cells above y=1
num_moving_cells_down = round(1 * num_cells_above_y1); % Number of cells to move down
moving_indices_down = cells_above_y1(randperm(num_cells_above_y1,num_moving_cells_down)); % Indices of cells to move down

% Identify cells below y=-1
cells_below_yn1 = find(pos(:,2) <= lower_bdry); % Cells below y=-1
num_cells_below_yn1 = length(cells_below_yn1); % Number of cells below y=-1

% Randomly select 50% of the cells below y=-1
num_moving_cells_up = round(1 * num_cells_below_yn1); % Number of cells to move up
moving_indices_up = cells_below_yn1(randperm(num_cells_below_yn1,num_moving_cells_up)); % Indices of cells to move up

tic % Start timing the simulation
for t = 1:Tmax/dt
    % Zero out forces
    F_cc = zeros(cells_max,2); % Cell-cell repulsive forces
    F_epid = zeros(cells_max,2); % Epidermis repulsive forces
    F_pull = zeros(cells_max,2); % Pulling forces
    F_active = zeros(Nb,2); % Active forces (unused)
    F_collision = zeros(length(Xb),2); % Collision forces
    v = zeros(cells_max,2); % Velocities

    if t==1
        % Plotting initially
        figure(1);
        scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','y','markeredgecolor','b'); % Plot cells
        hold on;
        plot(Xb(:,1), Xb(:,2), 'k','linewidth',2); % Plot boundary

        for i = 1:length(pos)
            if ~isnan(pos(i,1))
                if division_status(i)
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor',[0.3010 0.7450 0.9330],'markeredgecolor','b') % Blue for divided cells
                else
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor','y','markeredgecolor','b') % Yellow for original cells
                end
            end
        end 

        % Plot moving cells
        scatter(pos(moving_indices_down,1), pos(moving_indices_down,2),100,'o','MarkerFaceColor','r','MarkerEdgeColor','b'); % Red for moving down
        scatter(pos(moving_indices_up,1), pos(moving_indices_up,2),100,'o','MarkerFaceColor','m','MarkerEdgeColor','b'); % Magenta for moving up

        plot(xb,yb,'g.-') % Plot semi-circle
        yline(0) % Plot y=0 line
        yline(upper_bdry,'--') % Plot upper boundary
        yline(lower_bdry,'--') % Plot lower boundary

        xlim([-1.5 2])
        ylim([-2 2])

        xlabel('x')
        ylabel('y')
        title('Cell Positions at Time t = 0')
        hold off;
    end
    
    % Update the division interval
    next_div_time = div_interval; % Reset the next division time
    
    % Cell-cell repulsive forces
    for i = 1:N
        for j = 1:N
            if i ~= j && alive(i) && alive(j)
                rij = pos(i,:) - pos(j,:); % Vector between cells i and j
                d = norm(rij); % Distance between cells i and j
                if d < dl_crit
                    F_cc(i,:) = F_cc(i,:) + 0.01 * rij / d^2; % Repulsive force
                end
            end
        end
    end
    
    % Cell-boundary repulsive forces
    for i = 1:N
        for j = 1:Nb
            rij = pos(i,:) - Xb(j,:); % Vector between cell i and boundary point j
            d = norm(rij); % Distance between cell i and boundary point j
            if d < dl_crit
                F_epid(i,:) = F_epid(i,:) + kcoll * rij / d^2; % Repulsive force
            end
        end
    end
    
    % Pulling forces (connect cells to boundary points)
    for i = 1:N
        if alive(i)
            % Find the closest boundary point
            distances = sqrt(sum((Xb - pos(i,:)).^2, 2)); % Distances between cell i and all boundary points
            [~, closest_index] = min(distances); % Index of the closest boundary point
            rij = Xb(closest_index,:) - pos(i,:); % Vector between cell i and the closest boundary point
            d = norm(rij); % Distance between cell i and the closest boundary point
            F_pull(i,:) = F_pull(i,:) + kb * rij / d^2; % Pulling force
        end
    end

    % Compute cell velocities
    v = (F_cc + F_epid + F_pull) / xi;

    % Move cells
    pos(1:N,:) = pos(1:N,:) + v(1:N,:) * dt;

    % Cell division
    if mod(t, next_div_time/dt) == 0 && n_daughter < 2 * N
        % Choose a random cell to divide
        dividing_cell_index = randi(N);

        % Create a daughter cell
        mother_pos = pos(dividing_cell_index,:);
        daughter_pos = mother_pos + offset * (2*rand(1,2)-1); % Random offset
        pos(N+1,:) = daughter_pos;
        division_status(N+1) = true; % Mark the daughter cell as divided
        alive(N+1) = 1; % Mark the daughter cell as alive
        N = N + 1; % Increase the number of cells
        n_daughter = n_daughter + 1; % Increase the number of daughter cells
    end

    % Movement of selected cells
    pos(moving_indices_down,2) = pos(moving_indices_down,2) - 0.002; % Move cells down
    pos(moving_indices_up,2) = pos(moving_indices_up,2) + 0.002; % Move cells up

    % Check if any cells have moved out of the bounds
    cells_out_of_bounds = find(pos(:,2) > upper_bdry | pos(:,2) < lower_bdry);
    for i = cells_out_of_bounds'
        pos(i,:) = NaN; % Remove out-of-bounds cells
        alive(i) = 0; % Mark the cell as dead
    end

    % Update boundary points positions
    F_bdry = -kb * (Xb - Xb0); % Boundary spring force
    F_bdry_norm = sqrt(sum(F_bdry.^2, 2)); % Norm of the boundary force
    Xb = Xb + F_bdry .* dt; % Update boundary positions

    % Ensure the boundary points stay within a fixed distance
    for j = 1:Nb
        for k = j+1:Nb
            rij = Xb(j,:) - Xb(k,:); % Vector between boundary points j and k
            d = norm(rij); % Distance between boundary points j and k
            if d > 1.5 * dsb % If distance exceeds 1.5 times the initial distance
                Xb(j,:) = Xb(j,:) - 0.5 * rij * (d - 1.5 * dsb) / d; % Move boundary points closer
                Xb(k,:) = Xb(k,:) + 0.5 * rij * (d - 1.5 * dsb) / d; % Move boundary points closer
            end
        end
    end

    % Plot the cells and boundary points
    if mod(t,1.0/dt) == 0 % Plot every 1.0 time unit
        figure(1);
        scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','y','markeredgecolor','b'); % Plot cells
        hold on;
        plot(Xb(:,1), Xb(:,2), 'k','linewidth',2); % Plot boundary

        for i = 1:length(pos)
            if ~isnan(pos(i,1))
                if division_status(i)
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor',[0.3010 0.7450 0.9330],'markeredgecolor','b') % Blue for divided cells
                else
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor','y','markeredgecolor','b') % Yellow for original cells
                end
            end
        end 

        % Plot moving cells
        scatter(pos(moving_indices_down,1), pos(moving_indices_down,2),100,'o','MarkerFaceColor','r','MarkerEdgeColor','b'); % Red for moving down
        scatter(pos(moving_indices_up,1), pos(moving_indices_up,2),100,'o','MarkerFaceColor','m','MarkerEdgeColor','b'); % Magenta for moving up

        plot(xb,yb,'g.-') % Plot semi-circle
        yline(0) % Plot y=0 line
        yline(upper_bdry,'--') % Plot upper boundary
        yline(lower_bdry,'--') % Plot lower boundary

        xlim([-1.5 2])
        ylim([-2 2])

        xlabel('x')
        ylabel('y')
        title(['Cell Positions at Time t = ', num2str(t*dt)])
        hold off;
        
        % if(video_flag)
        %     currFrame = getframe(gcf); % Get the current frame
        %     writeVideo(vidObj,currFrame); % Write the frame to the video object
        % end
    end
end
toc % Stop timing the simulation

if(video_flag)
    close(vidObj); % Close the video object
end
