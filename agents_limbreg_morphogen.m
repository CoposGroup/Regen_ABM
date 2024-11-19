% Agent-based stochastic model
% Cells represented as point particles
% Brownian motion and repulsion
%
% CC (Jan 2024)


close all;
% clc;
clear;


load('cellinitialization_n500','pos0','Ncells');
N = Ncells;
% a = 0; b = 2*pi; th0 = (b-a).*rand(N,1) + a;
% r_circle    = 1.35;
% pos0        = [r_circle*rand(N,1).*cos(th0),r_circle*rand(N,1).*sin(th0)];
n = length(pos0);

video_flag = 0;
if(video_flag)
    vidObj = VideoWriter('morphogen.mp4','MPEG-4');
    open(vidObj);
end

% Parameters
dt          = 0.00001;
Tmax        = 5.0;
mu          = 1;
dl_crit     = 0.1; % critical distance for computing repulsive force
xi          = 1.0; % drag coefficient
kb          = 10.0; % stiffness of epidermis
kcoll       = 0.08; % epidermins-cell collision constant
kdiv        = 0.1; % proliferation rate
offset      = 0.1; % distance between mother and daughter cell

pos     = pos0;
pre_pos = pos;

% A large array for positions
cells_max   = 5 * N;
pos         = NaN(cells_max,2);
v           = NaN(cells_max,2);
tau         = NaN(cells_max,1);
v(1:N,:)    = zeros(N,2);
tau(1:N,:)  = zeros(N,1);
pos(1:N, :) = pos0;

% Division status
division_status = false(cells_max, 1);

% Division interval
div_interval = 0.4; 
next_div_time = div_interval;

% Define semi-circle
semi_circle_center = [0, 0];
semi_circle_radius = 1.5;
theta = linspace(3*pi/2,5*pi/2);
xb_semi_circle = semi_circle_radius * cos(theta) + semi_circle_center(1);
yb_semi_circle = semi_circle_radius * sin(theta) + semi_circle_center(2);

% Define the vertical line segments
x_vert = [1.2, 1.2];

diffs = abs(xb_semi_circle - x_vert(1));
[~, sorted_indices] = sort(diffs);
closest_indices = sorted_indices(1:2); % find the indices of the two smallest differences
val_yb = yb_semi_circle(closest_indices);
val_xb = xb_semi_circle(closest_indices);

y_vert = [val_yb(1),val_yb(2)];

% Find mean spacing
xb_space = xb_semi_circle(1:min(closest_indices));
yb_space = yb_semi_circle(1:min(closest_indices));
for i=1:length(xb_space)-1
    ds = sqrt((xb_space(i+1)-xb_space(i))^2+(yb_space(i+1)-yb_space(i))^2);
end
avg_ds = mean(ds);

y_v = y_vert(2):avg_ds:y_vert(1);
x_v = 1.2*ones(1,length(y_v));


% Combine the semicircle and the vertical line segment in correct order
xb = [xb_semi_circle(1:min(closest_indices)),x_v,xb_semi_circle(max(closest_indices):end)];
yb = [yb_semi_circle(1:min(closest_indices)),y_v,yb_semi_circle(max(closest_indices):end)];

Xb0 = [xb;yb]'; 
Xb = [xb;yb]';
dsb = sqrt( (Xb0(1,1)-Xb0(2,1))^2 + (Xb0(1,2)-Xb0(2,2))^2 ); % initial distance btw the first 2 pts on the b

% Make a difference matrix for the boundary points
Nb = length(Xb);
e = ones(Nb,1);
Db = spdiags([-e e],[0  1],Nb,Nb); % cal the diff in pos of adjacent pts along the b
%Db(1,1) = 1; Db(1,2) = 0;
Db(Nb,1) = 1;

% Tether points/top channel resting length
blp0 = Db * Xb0;
blp0 = sqrt(blp0(:,1).^2 + blp0(:,2).^2); % rest length of edge from i to i+1
blm0 = Db' * Xb0;
blm0 = sqrt(blm0(:,1).^2 + blm0(:,2).^2); % rest length of edge from i-1 to i+1

% Alive status
alive = NaN(length(pos),1);
not_NaN_rows = ~any(isnan(pos),2);
alive(not_NaN_rows & isnan(alive)) = 1;

% Number of daughter cells 
n_daughter = 0;





tic
for t = 1:Tmax/dt
    % Zero out 
    F_cc = zeros(cells_max,2);
    F_epid = zeros(cells_max,2);
    F_pull = zeros(cells_max,2);
    F_active = zeros(Nb,2);
    F_collision = zeros(length(Xb),2);
    v = zeros(cells_max,2);


    if t==1
        % Plotting initially
        figure(1);
        scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','y','markeredgecolor','b'); 
        hold on;
        plot(Xb(:,1), Xb(:,2), 'k','linewidth',2);

        for i = 1:length(pos)
            if ~isnan(pos(i,1))
                if division_status(i)
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor',[0.3010 0.7450 0.9330],'markeredgecolor','b') % blue for divided cells
                else
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor','y','markeredgecolor','b') % yellow for original cells
                end
            end
        end 

        plot(xb,yb,'r.-')
        yline(1.5,'--')
        yline(1,'--')
        yline(0.5,'--')
        yline(0,'--')



        % Fill the area between y=0 and y=0.5
        x_0_0pt5 = [0 1.2 1.2 0];
        y_0_0pt5 = [0 0 0.5 0.5];
        fill(x_0_0pt5,y_0_0pt5,'r','FaceAlpha',0.3)



        % Fill the area between y=0.5 and y=1
        % find y value closest to 1, and its corresponding x value
        [~,idx_y1] = min(abs(Xb(:,2) - 1));
        y_1 = Xb(idx_y1,2);
        y_1_x = Xb(idx_y1,1);

        % find Xb values between y=0.5 and y=1
        idx_y0pt5_y1 = (Xb(:,2) >= 0.5) & (Xb(:,2) <= 1);
        Xb_y0pt5_y1 = Xb(idx_y0pt5_y1,:);

        x_0pt5_1 = [0; 1.2; Xb_y0pt5_y1(:,1); y_1_x; 0];
        y_0pt5_1 = [0.5; 0.5; Xb_y0pt5_y1(:,2); y_1; 1];
        fill(x_0pt5_1,y_0pt5_1,'b','FaceAlpha',0.3)



        % Fill the area between y=1 and y=1.5
        % find Xb values between y=1 and y=1.5
        idx_y1_y1pt5 = (Xb(:,2) >= 1) & (Xb(:,2) <= 1.5);
        Xb_y1_y1pt5 = Xb(idx_y1_y1pt5,:);

        x_1_1pt5 = [0; y_1_x; Xb_y1_y1pt5(:,1); 0];
        y_1_1pt5 = [1; y_1; Xb_y1_y1pt5(:,2); 1.5];
        fill(x_1_1pt5,y_1_1pt5,'c','FaceAlpha',0.3)





        xlim([-1.5 2.5]); ylim([-1.5 1.5]); box on; grid off;
        set(gca,'plotBoxAspectRatio',[1 1 1]);
        set(gca,'FontSize',20);% set(gca,'Color','k');
        set(gcf,'color','w');% set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
        x0=800;y0=700;width=400;height=400;
        set(gcf,'position',[x0,y0,width,height]);

        currFrame = getframe(gcf);
        % writeVideo(vidObj,currFrame);

        pause(0.2)
        hold off;
    end


    % Cell division/apoptosis
    active_cells = find(~isnan(pos(:,1)));
    N_active = length(active_cells);
    r = rand;
    if r < (N_active-n_daughter)*kdiv*dt
        % Randomly pick dividing cell
        dividing_cell = active_cells(randperm(length(active_cells), 1));
        N_active = N_active + 1;
        n_daughter = n_daughter + 1;

        new_idx = find(isnan(pos(:,1)), 1, 'first'); % daughter cell

        % % Position of daughter cell (random direction)
        % div_angle = 2*pi*rand;
        % dx = offset * cos(div_angle);
        % dy = offset * sin(div_angle);
        % new_pos = pos(dividing_cell,:) + [dx, dy];

        % Position of daughter cell (directed)
        div_angle = 0 - pi/6*randn;
        dx = offset * cos(div_angle);
        dy = offset * sin(div_angle);
        new_pos = pos(dividing_cell,:) + [dx, dy];

        pos(new_idx,:) = new_pos;
        alive(new_idx,:) = 1;
        division_status(new_idx) = true;
    end


    % For active cells: compute forces & update positions
    active_cells = find(~isnan(pos(:,1)));
    N_active_cells = length(active_cells);

    F_cc(active_cells,:) = compute_cellcell_repulsive(pos(active_cells,:),dl_crit);
    F_epid(active_cells,:) = compute_semi_circle_repulsion(pos(active_cells,:), dl_crit, Xb);
    % F_pull(active_cells,:) = [ones(N_active_cells,1),zeros(N_active_cells,1)];




    % Identify cells with different conditions
    cells_below_y0 = find(pos(:,2) < 0); % cells below y=0
    cells_y_0_0p5 = find(pos(:,2) >= 0 & pos(:,2) < 0.5); % cells between y=0 and y=0.5
    cells_y_0p5_1 = find(pos(:,2) >= 0.5 & pos(:,2) < 1); % cells between y=0.5 and y=1
    cells_y_1_1p5 = find(pos(:,2) >= 1 & pos(:,2) <= 1.5); % cells between y=1 and y=1.5

    % Apply pull force 
    F_pull(cells_below_y0,:) = [zeros(length(cells_below_y0),1),zeros(length(cells_below_y0),1)];
    F_pull(cells_y_0_0p5,:) = [ones(length(cells_y_0_0p5),1) * 0.5,zeros(length(cells_y_0_0p5),1)];
    F_pull(cells_y_0p5_1,:) = [ones(length(cells_y_0p5_1),1),zeros(length(cells_y_0p5_1),1)];
    F_pull(cells_y_1_1p5,:) = [ones(length(cells_y_1_1p5),1) * 2,zeros(length(cells_y_1_1p5),1)];

 


    % Apply forces to the boundary
    for i = 1:length(active_cells)
        if norm(F_epid(i,:)) > 0 % indicates collision with boundary
            [Fpos,mink_ids] = compute_boundary_collision_force(pos(active_cells(i),:),Xb,kcoll);
            F_collision(mink_ids,:) = Fpos;
        end
    end

    F = F_cc + 5*F_epid + F_pull;

    v = F/xi;
    a = -6; b = 6; eta = (b-a).*rand(length(active_cells),2) + a; % Brownian motion
    pos(active_cells,:) = pos(active_cells,:) + v(active_cells,:)*dt + dt*eta*6;

    % For semi-circle (calling it "b" for boundary): compute forces & update positions
    upper_indices = find(yb >= 0);
    rest_indices = find(~(yb >= 0));
    F_semi_circle_elasticity = compute_semi_circle_elasticity(Xb, Db, blp0, blm0, dsb, kb, upper_indices, rest_indices);
    %F_semi_circle_interp = interp1(pos,F_semi_circle,Xb);

    F_active = zeros(Nb,2);
    %F_active(40:60,1) = 1;

    Fb = F_semi_circle_elasticity + 5*F_collision;%+ F_semi_circle_interp;
    Xb(2:end-1,:) = Xb(2:end-1,:) + dt*(Fb(2:end-1,:)/xi);


    if mod(t,10000) == 0
        % Plotting
        figure(1);
        scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','y','markeredgecolor','b'); 
        hold on;
        plot(Xb(:,1), Xb(:,2), 'k','linewidth',2);

        for i = 1:length(pos)
            if ~isnan(pos(i,1))
                if division_status(i)
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor',[0.3010 0.7450 0.9330],'markeredgecolor','b') % blue for divided cells
                else
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor','y','markeredgecolor','b') % yellow for original cells
                end
            end
        end 




        plot(xb,yb,'r.-')
        yline(1.5,'--')
        yline(1,'--')
        yline(0.5,'--')
        yline(0,'--')





        % Fill the area between y=0 and y=0.5
        % find y value closest to 0, and its corresponding x value
        [~,idx_y0] = min(abs(Xb(:,2)));
        y_0 = Xb(idx_y0,2);
        y_0_x = Xb(idx_y0,1);

        % find y value closest to 0.5, and its corresponding x value
        [~,idx_ypt5] = min(abs(Xb(:,2) - 0.5));
        y_0pt5 = Xb(idx_ypt5,2);
        y_0pt5_x = Xb(idx_ypt5,1);

        % find Xb values between y=0 and y=0.5
        idx_y0_y0pt5 = (Xb(:,2) >= 0) & (Xb(:,2) <= 0.5);
        Xb_y0_y0pt5 = Xb(idx_y0_y0pt5,:);

        x_0_0pt5 = [0; y_0_x; Xb_y0_y0pt5(:,1); y_0pt5_x; 0];
        y_0_0pt5 = [0; y_0; Xb_y0_y0pt5(:,2); y_0pt5; 0.5];
        fill(x_0_0pt5,y_0_0pt5,'r','FaceAlpha',0.3)



        % Fill the area between y=0.5 and y=1
        % find y value closest to 1, and its corresponding x value
        [~,idx_y1] = min(abs(Xb(:,2) - 1));
        y_1 = Xb(idx_y1,2);
        y_1_x = Xb(idx_y1,1);

        % find Xb values between y=0.5 and y=1
        idx_y0pt5_y1 = (Xb(:,2) >= 0.5) & (Xb(:,2) <= 1);
        Xb_y0pt5_y1 = Xb(idx_y0pt5_y1,:);

        x_0pt5_1 = [0; y_0pt5_x; Xb_y0pt5_y1(:,1); y_1_x; 0];
        y_0pt5_1 = [0.5; y_0pt5; Xb_y0pt5_y1(:,2); y_1; 1];
        fill(x_0pt5_1,y_0pt5_1,'b','FaceAlpha',0.3)



        % Fill the area between y=1 and y=1.5
        % find Xb values between y=1 and y=1.5
        idx_y1_y1pt5 = (Xb(:,2) >= 1) & (Xb(:,2) <= 1.5);
        Xb_y1_y1pt5 = Xb(idx_y1_y1pt5,:);

        x_1_1pt5 = [0; y_1_x; Xb_y1_y1pt5(:,1); 0];
        y_1_1pt5 = [1; y_1; Xb_y1_y1pt5(:,2); 1.5];
        fill(x_1_1pt5,y_1_1pt5,'c','FaceAlpha',0.3)







        % quiver(pos(:,1),pos(:,2),F_cc(:,1),F_cc(:,2),'r','linewidth',2,'autoscale','off');
        % quiver(pos(:,1),pos(:,2),F_epid(:,1),F_epid(:,2),'b','linewidth',2,'autoscale','off');
        % quiver(Xb(:,1),Xb(:,2),F_collision(:,1),F_collision(:,2),'-w','autoscale','on')
        % scatter(Xb(mink_ids,1),Xb(mink_ids,2),'g*')

        xlim([-1.5 2.5]); ylim([-1.5 1.5]); box on; grid off;

        set(gca,'plotBoxAspectRatio',[1 1 1]);
        set(gca,'FontSize',20);% set(gca,'Color','k');
        set(gcf,'color','w');% set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
        x0=800;y0=700;width=400;height=400;
        set(gcf,'position',[x0,y0,width,height]);

        currFrame = getframe(gcf);
        % writeVideo(vidObj,currFrame);

        pause(0.2)
        hold off;



        % % Histogram for probability distribution
        % figure(2);
        % 
        % blue_cells = pos(division_status,:);
        % yellow_cells = pos(~division_status,:);
        % 
        % % remove NaN
        % blue_cells = blue_cells(all(~isnan(blue_cells),2),:);
        % yellow_cells = yellow_cells(all(~isnan(yellow_cells),2),:);
        % 
        % histogram(blue_cells(:,1),'BinWidth',0.1,'Normalization','probability','FaceColor','b');
        % xlim([-2 2])
        % hold on;
        % histogram(yellow_cells(:,1),'BinWidth',0.1,'Normalization','probability','FaceColor','y');
        % 
        % legend('Blue (Divided) Cells', 'Yellow (Original) Cells');
        % currFrame = getframe(gcf);
        % hold off;

    end


end
toc


if(video_flag)
    close(vidObj);
end



function [F_repulsive] = compute_cellcell_repulsive(pos,dl_crit)
    F_repulsive = zeros(length(pos),2);
    n = length(pos);

    for i=1:n
        for j=1:n
            dl = sqrt( (pos(i,1)-pos(j,1))^2 + (pos(i,2)-pos(j,2))^2 );
            if (j~=i)&&(dl<dl_crit)
                dl = sqrt( (pos(i,1)-pos(j,1))^2 + (pos(i,2)-pos(j,2))^2 );
                F_repulsive(j,:) = -2*(pos(i,:)-pos(j,:))/dl;
                F_repulsive(i,:) = -2*(pos(j,:)-pos(i,:))/dl;
            end
        end
    end

end




function [F_semi_circle] = compute_semi_circle_repulsion(pos, dl_crit, Xb)
    n = size(pos, 1);
    F_semi_circle = zeros(n, 2);

    for i = 1:n
        cell_pos = pos(i, :);

        for j = 1:length(Xb)
            r = sqrt((cell_pos(1) - Xb(j,1))^2 + (cell_pos(2) - Xb(j,2))^2);
            if r < dl_crit
                F_semi_circle(i, :) = F_semi_circle(i, :) + [-0.5, 0];
                %F_semi_circle(i, :)
                %sprintf('Collision!')
                %keyboard()
            end
        end
    end
end





function[F_deformable_semi_circle] = compute_semi_circle_elasticity(Xb, Db, blp0, blm0, dsb, kb, mid_part_indices, rest_indices)
    Nb = length(Xb);
    F_deformable_semi_circle = zeros(Nb,2);

    kb_values = kb * ones(Nb, 1);
    kb_values(mid_part_indices) = kb;%kb/1000;

    % Compute spring forces between nodes
    btem_p = Db * Xb;
    blp = sqrt(btem_p(:,1).^2 + btem_p(:,2).^2); % length of edge from i to i+1
    btem_m = Db' * Xb;
    blm = sqrt(btem_m(:,1).^2 + btem_m(:,2).^2); % length of edge from i-1 to i
    Fbs = (kb_values .* ( blp./blp0 - 1. )) .* ((Db * Xb)./blp) + ...
          (kb_values .* ( blm./blm0 - 1. )) .* ((Db'*Xb)./blm);
    Fbs = Fbs ./ dsb;

    F_deformable_semi_circle = F_deformable_semi_circle + Fbs; % add spring forces to tethering forces
end




function [Fpos,mink_ids] = compute_boundary_collision_force(cell_pos, Xb, kcoll)
    % Calculate the distances between cell_pos and boundary points
    distances = sqrt( (Xb(:,1)-cell_pos(1)).^2 + (Xb(:,2)-cell_pos(2)).^2 );
    
    % Find the nearest point
    [~,min_idx] = mink(distances,3);
    nearest_boundary_points = Xb(min_idx,:);

    % Calculate the force
    mink_distances = sqrt((cell_pos(1) - nearest_boundary_points(:,1)).^2 + (cell_pos(2) - nearest_boundary_points(:,2)).^2);
    directions = (nearest_boundary_points - cell_pos) ./ mink_distances; % unit vector
    Fpos = kcoll * (mink_distances./sum(mink_distances)) .* directions;
    mink_ids = min_idx;

end









