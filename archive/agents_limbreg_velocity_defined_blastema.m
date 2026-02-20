% Agent-based stochastic model
% Velocity Test
% Epidermis is defined as line 25-59
%
% SW (JUN 2024)


close all;
% clc;
clear;


video_flag = 0;
if(video_flag)
    vidObj = VideoWriter('test.mp4','MPEG-4');
    open(vidObj);
end


% Parameters
dt          = 0.00001;
Tmax        = 5.0;


% Define semi-circle
semi_circle_center = [0, 0];
semi_circle_radius = 1.5;
theta = linspace(3*pi/2,5*pi/2,1027);
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

y_v = y_vert(1):avg_ds:y_vert(2);
x_v = 1.2*ones(1,length(y_v));

% Combine the semicircle and the vertical line segment in correct order
xb = [xb_semi_circle(1:min(closest_indices)),x_v,xb_semi_circle(max(closest_indices):end)];
yb = [yb_semi_circle(1:min(closest_indices)),y_v,yb_semi_circle(max(closest_indices):end)];

Xb0 = [xb;yb]'; 
Xb = [xb;yb]';


% Load velocity data
load('CTRL_Velocity.mat','v_2_12_CTRL_m');



% Rotation matrix
theta = pi/2;
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
Xb_rotated = Xb * R;

Xb0_rotated = Xb0 * R;


tic
for t = 1:Tmax/dt


    Xb_rotated = Xb_rotated + dt*(v_2_12_CTRL_m);
    

    if mod(t,100) == 0
        % plotting
        figure(1);
        plot(Xb_rotated(:,1), Xb_rotated(:,2), 'k','linewidth',2);

        hold on;

        plot(Xb0_rotated(:,1),Xb0_rotated(:,2),'r','LineWidth',2);

        % quiver(Xb_rotated(:,1),Xb_rotated(:,2),v_2_12_CTRL_m(:,1),v_2_12_CTRL_m(:,2),'-r');
        xlim([-12 12]); ylim([-12 0.5]); box on; grid off;
        set(gca,'plotBoxAspectRatio',[1 1 1]);
        set(gca,'FontSize',10); %set(gca,'Color','k')
        set(gcf,'color','w'); %set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
        x0=800;y0=700;width=400;height=400;
        set(gcf,'position',[x0,y0,width,height]);

        currFrame = getframe(gcf);
        % writeVideo(vidObj,currFrame);

        pause(0.2)
        hold off;

    end

end
toc


if(video_flag)
    close(vidObj);
end


