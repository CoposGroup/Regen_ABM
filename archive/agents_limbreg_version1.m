% Agent-based stochastic model
% Cells represented as point particles
% Brownian motion and repulsion
% 
% Random Cell Movements + Epidermis
%
% CC (Jan 2024)



close all;
clc;
clear;


N = 20;
a = 0; b = 2*pi; th0 = (b-a).*rand(N,1) + a;
r_circle    = 1.5;
pos0        = [r_circle*rand(N,1).*cos(th0),r_circle*rand(N,1).*sin(th0)];
n = length(pos0);


video_flag=0;
if(video_flag)
    vidObj = VideoWriter('test.mp4','MPEG-4');
    open(vidObj);
end


% parameters
dt          = 0.01;
Tmax        = 2.0;
mu          = 1;
dl_crit     = 0.1; % critical distance for computing repulsive force
xi          = 1.0; % drag coefficient

pos = pos0;
v = zeros(n,2);
pre_pos = pos;

% a large array for positions
cells_max = 2 * N;
pos = NaN(cells_max, 2);
pos(1:N, :) = pos0;

% division status
division_status = false(cells_max, 1);

% division interval
div_interval = 0.3; 
next_div_time = div_interval;

offset = 0.1; % distance between mother and daughter cell

% define semi-circle
semi_circle_center = [0, 0];
semi_circle_radius = 1.5;


for t = 1:Tmax/dt
    % cell - repulsion force
    F_repulsion = compute_repulsive(pos,dl_crit);


    % F_pull = [ones(N,1),zeros(N,1)];


    % plotting
    figure(3);
    scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','y','markeredgecolor','b'); 
    hold on;


    % semi-circle
    theta = linspace(3*pi/2,5*pi/2);
    x = semi_circle_radius * cos(theta) + semi_circle_center(1);
    y = semi_circle_radius * sin(theta) + semi_circle_center(2);
    plot(x, y, 'w','linewidth',2);
    


    for i = 1:length(pos)
        if ~isnan(pos(i,1))
            if division_status(i)
                scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor',[0.3010 0.7450 0.9330],'markeredgecolor','b')
            else
                scatter(pos(i,1),pos(i,2),100,'o','markerfacecolor','y','markeredgecolor','b')
            end
        end
    end % yellow for original cells, blue for divided cells


    quiver(pos(:,1),pos(:,2),0.1*F_repulsion(:,1),0.1*F_repulsion(:,2),'r','linewidth',2,'autoscale','off');

    xlim([-1.5 2.5]); ylim([-1.5 1.5]); box on; grid off;

    set(gca,'plotBoxAspectRatio',[1 1 1]);
    set(gca,'FontSize',20); set(gca,'Color','k')
    set(gcf,'color','w'); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    x0=800;y0=700;width=400;height=400;
    set(gcf,'position',[x0,y0,width,height]);
    currFrame = getframe(gcf);
    pause(0.2)
    hold off;



    % cell division
    if t*dt >= next_div_time

        % Randomly select 50% of the cells for division
        active_cells = find(~isnan(pos(:,1)));
        dividing_cells = active_cells(randperm(length(active_cells), length(active_cells)*0.5));

        for i = 1:length(dividing_cells)
            cell_idx = dividing_cells(i); % mother cell
            new_idx = find(isnan(pos(:,1)), 1, 'first'); % daughter cell

            % position of daughter cell
            div_angle = 0 + pi/6*randn;
            dx = offset * cos(div_angle);
            dy = offset * sin(div_angle);
            new_pos = pos(cell_idx,:) + [dx, dy];

            pos(new_idx,:) = new_pos;

            division_status(new_idx) = true;


            % keyboard()
            % hold on; scatter(pos(new_idx,1),pos(new_idx,2),100,'ro','filled'); div_angle

            next_div_time = next_div_time + div_interval;

        end

        
    end

    % for active cells
    active_cells = find(~isnan(pos(:,1)));

    F_repulsion = compute_repulsive(pos(active_cells,:),dl_crit);
    F_semi_circle = compute_semi_circle_repulsion(pos(active_cells,:), dl_crit, semi_circle_center, semi_circle_radius);

    F = F_repulsion + F_semi_circle;

    v(active_cells,:) = F/xi;
    a = -2; b = 2; eta = (b-a).*rand(length(active_cells),2) + a; % Brownian motion
    pos(active_cells,:) = pos(active_cells,:) + v(active_cells,:)*dt + dt*eta;


    if(video_flag)
        writeVideo(vidObj,currFrame);
    end
end


if(video_flag)
    close(vidObj);
end


function [F_repulsive] = compute_repulsive(pos,dl_crit)
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


function [F_semi_circle] = compute_semi_circle_repulsion(pos, dl_crit, semi_circle_center, semi_circle_radius)
    n = size(pos, 1);
    F_semi_circle = zeros(n, 2);

    theta = linspace(3*pi/2, 5*pi/2);
    semi_circle_x = semi_circle_radius * cos(theta) + semi_circle_center(1);
    semi_circle_y = semi_circle_radius * sin(theta) + semi_circle_center(2);

    for i = 1:n
        cell_pos = pos(i, :);

        for j = 1:length(theta)
            r = sqrt((cell_pos(1) - semi_circle_x(j))^2 + (cell_pos(2) - semi_circle_y(j))^2);
            if r < dl_crit
                F_semi_circle(i, :) = F_semi_circle(i, :) + [-0.1, 0];
                break; % Break the loop if the force is applied
            end
        end
    end
end





