% Agent-based stochastic model
% Cells represented as point particles
% Brownian motion and repulsion
%
% CC (Jan 2024)

close all;
% clc;
clear;


load('cellinitialization_n100','pos0','Ncells');


N = Ncells;
% a = 0; b = 2*pi; th0 = (b-a).*rand(N,1) + a;
% r_circle    = 1.35;
% pos0        = [r_circle*rand(N,1).*cos(th0),r_circle*rand(N,1).*sin(th0)];
n = length(pos0);

video_flag = 0;
if(video_flag)
    vidObj = VideoWriter('test.mp4','MPEG-4');
    open(vidObj);
end

% parameters
dt          = 0.00001;
Tmax        = 5.0;
mu          = 1;
dl_crit     = 0.1; % critical distance for computing repulsive force
xi          = 1.0; % drag coefficient
kb          = 10.0; % stiffness of epidermis
kcoll       = 0.08; % epidermins-cell collision constant

pos = pos0;
v = zeros(n,2);
pre_pos = pos;

% a large array for positions
cells_max = 5 * N;
pos = NaN(cells_max,2);
pos(1:N,:) = pos0;

% division status
division_status = false(cells_max, 1);

% division interval
div_interval = 0.4; 
next_div_time = div_interval;

offset = 0.1; % distance between mother and daughter cell

% define semi-circle
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

% find mean spacing
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

% make a difference matrix for the boundary points
Nb = length(Xb);
e = ones(Nb,1);
Db = spdiags([-e e],[0  1],Nb,Nb); % cal the diff in pos of adjacent pts along the b
%Db(1,1) = 1; Db(1,2) = 0;
Db(Nb,1) = 1;

% tether points/top channel resting length
blp0 = Db * Xb0;
blp0 = sqrt(blp0(:,1).^2 + blp0(:,2).^2); % rest length of edge from i to i+1
blm0 = Db' * Xb0;
blm0 = sqrt(blm0(:,1).^2 + blm0(:,2).^2); % rest length of edge from i-1 to i+1



% alive status
alive = NaN(length(pos),1);
not_NaN_rows = ~any(isnan(pos),2);
alive(not_NaN_rows & isnan(alive)) = 1;




tic
for t = 1:Tmax/dt
    % zero out 
    F_cc = zeros(cells_max,2);
    F_epid = zeros(cells_max,2);
    F_pull = zeros(cells_max,2);
    F_active = zeros(Nb,2);
    F_collision = zeros(length(Xb),2);
    v = zeros(cells_max,2);


    if t==1
        % plotting initially
        figure(1);
        scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','y','markeredgecolor','b'); 
        hold on;
        plot(Xb(:,1), Xb(:,2), 'w','linewidth',2);

        for i = 1:length(pos)
            if ~isnan(pos(i,1))
                if division_status(i)
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor',[0.3010 0.7450 0.9330],'markeredgecolor','b') % blue for divided cells
                else
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor','y','markeredgecolor','b') % yellow for original cells
                end
            end
        end 


        plot(xb(40:60),yb(40:60),'r','LineWidth',3);


        xlim([-1.5 2.5]); ylim([-1.5 1.5]); box on; grid off;
        set(gca,'plotBoxAspectRatio',[1 1 1]);
        set(gca,'FontSize',20); set(gca,'Color','k')
        set(gcf,'color','w'); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
        x0=800;y0=700;width=400;height=400;
        set(gcf,'position',[x0,y0,width,height]);

        currFrame = getframe(gcf);
        % writeVideo(vidObj,currFrame);

        pause(0.2)
        hold off;
    end








    % cell division
    if t*dt >= next_div_time

        % Randomly select 50% of the cells for division
        active_cells = find(~isnan(pos(:,1)));
        dividing_cells = active_cells(randperm(length(active_cells), round(length(active_cells)*0.5)));

        for i = 1:length(dividing_cells)
            cell_idx = dividing_cells(i); % mother cell
            new_idx = find(isnan(pos(:,1)), 1, 'first'); % daughter cell


            % % position of daughter cell (random direction)
            % div_angle = 2*pi*rand;
            % dx = offset * cos(div_angle);
            % dy = offset * sin(div_angle);
            % new_pos = pos(cell_idx,:) + [dx, dy];


            % position of daughter cell (directed)
            div_angle = 0 - pi/6*randn;
            dx = offset * cos(div_angle);
            dy = offset * sin(div_angle);
            new_pos = pos(cell_idx,:) + [dx, dy];

            pos(new_idx,:) = new_pos;
            alive(new_idx,:) = 1;

            division_status(new_idx) = true;

            next_div_time = next_div_time + div_interval;

        end


    end



    % For active cells: compute forces & update positions
    active_cells = find(~isnan(pos(:,1)));
    N_active_cells = length(active_cells);

    F_cc(active_cells,:) = compute_cellcell_repulsive(pos(active_cells,:),dl_crit);
    F_epid(active_cells,:) = compute_semi_circle_repulsion(pos(active_cells,:), dl_crit, Xb);
    F_pull(active_cells,:) = [ones(N_active_cells,1),zeros(N_active_cells,1)];

 
    % apply forces to the boundary
    for i = 1:length(active_cells)
        if norm(F_epid(i,:)) > 0 % indicates collision with boundary
            [Fpos,mink_ids] = compute_boundary_collision_force(pos(active_cells(i),:),Xb,kcoll);
            F_collision(mink_ids,:) = Fpos;
        end
    end

    F = F_cc + 5*F_epid + F_pull;

    v = F/xi;
    a = -2; b = 2; eta = (b-a).*rand(length(active_cells),2) + a; % Brownian motion
    pos(active_cells,:) = pos(active_cells,:) + v(active_cells,:)*dt + dt*eta;

    % For semi-circle (calling it "b" for boundary): compute forces & update positions
    mid_part_indices = find(yb >= -0.5 & yb <= 0.5);
    F_semi_circle_elasticity = compute_semi_circle_elasticity(Xb, Db, blp0, blm0, dsb, kb, mid_part_indices, semi_circle_center);
    %F_semi_circle_interp = interp1(pos,F_semi_circle,Xb);

    F_active = zeros(Nb,2);
    %F_active(40:60,1) = 1;

    Fb = F_semi_circle_elasticity + 5*F_collision;%+ F_semi_circle_interp;
    Xb(2:end-1,:) = Xb(2:end-1,:) + dt*(Fb(2:end-1,:)/xi);

    % if(video_flag)
    %     writeVideo(vidObj,currFrame);
    % end


    if mod(t,10000) == 0
        % plotting
        figure(1);
        scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','y','markeredgecolor','b'); 
        hold on;
        plot(Xb(:,1), Xb(:,2), 'w','linewidth',2);

        for i = 1:length(pos)
            if ~isnan(pos(i,1))
                if division_status(i)
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor',[0.3010 0.7450 0.9330],'markeredgecolor','b') % blue for divided cells
                else
                    scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor','y','markeredgecolor','b') % yellow for original cells
                end
            end
        end 


        plot(xb,yb,'g.-')
        plot(xb(40:60),yb(40:60),'r','LineWidth',3)

        %quiver(pos(:,1),pos(:,2),F_cc(:,1),F_cc(:,2),'r','linewidth',2,'autoscale','off');
        %quiver(pos(:,1),pos(:,2),F_epid(:,1),F_epid(:,2),'b','linewidth',2,'autoscale','off');
        % quiver(Xb(:,1),Xb(:,2),F_collision(:,1),F_collision(:,2),'-w','autoscale','on')
        % scatter(Xb(mink_ids,1),Xb(mink_ids,2),'g*')

        xlim([-1.5 2.5]); ylim([-1.5 1.5]); box on; grid off;

        set(gca,'plotBoxAspectRatio',[1 1 1]);
        set(gca,'FontSize',20); set(gca,'Color','k')
        set(gcf,'color','w'); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
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





function[F_deformable_semi_circle] = compute_semi_circle_elasticity(Xb, Db, blp0, blm0, dsb, kb, mid_part_indices, semi_circle_center)
    Nb = length(Xb);
    F_deformable_semi_circle = zeros(Nb,2);

    kb_values = kb * ones(Nb, 1);
    kb_values(mid_part_indices) = kb/1000;

    % compute spring forces between nodes
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
    % % Calculate the nearest pt on the bdry to the cell
    % % define y range
    % y_min = cell_pos(2) - 1.5;
    % y_max = cell_pos(2) + 1.5;
    % 
    % % find the points in Xb within the y-range
    % indices = Xb(:,2) >= y_min & Xb(:,2) <= y_max;
    % pts = Xb(indices,:);

    % calculate the distances between cell_pos and boundary points
    distances = sqrt( (Xb(:,1)-cell_pos(1)).^2 + (Xb(:,2)-cell_pos(2)).^2 );
    
    % find the nearest point
    [~,min_idx] = mink(distances,3);
    nearest_boundary_points = Xb(min_idx,:);

    % Calculate the force
    mink_distances = sqrt((cell_pos(1) - nearest_boundary_points(:,1)).^2 + (cell_pos(2) - nearest_boundary_points(:,2)).^2);
    directions = (nearest_boundary_points - cell_pos) ./ mink_distances; % unit vector
    %keyboard()
    Fpos = kcoll * (mink_distances./sum(mink_distances)) .* directions;
    mink_ids = min_idx;

end









