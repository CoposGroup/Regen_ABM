% Agent-based stochastic model
% Cells represented as point particles
% Brownian motion and repulsion
%
% CC (Jan 2024)

close all;
clc;
clear;

N = 20;
a = 0; b = 2*pi; th0 = (b-a).*rand(N,1) + a;
r_circle    = 0.5;
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
pre_pos = pos; % previous position

for t = 1:Tmax/dt
    % cell - repulsion force
    F_repulsion = compute_repulsive(pos,dl_crit);

    % pulling forces
    Fpull = [ones(N,1),-1*ones(N,1)];

    % plotting
    figure(3);
    scatter(pos(:,1),pos(:,2),100,'o','markerfacecolor','w','markeredgecolor','b'); 
    hold on;
    quiver(pos(:,1),pos(:,2),0.1*F_repulsion(:,1),0.1*F_repulsion(:,2),'r','linewidth',2,'autoscale','off');
    xlim([-1.0 1.0]); ylim([-1.0 1.0]); box on; grid off;
    set(gca,'plotBoxAspectRatio',[1 1 1]);
    set(gca,'FontSize',20); set(gca,'Color','k')
    set(gcf,'color','w'); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    x0=800;y0=700;width=400;height=400;
    set(gcf,'position',[x0,y0,width,height]);
    currFrame = getframe(gcf);
    pause(0.2)
    hold off;
 
    % sum of all forces
    F = F_repulsion + Fpull;

    % update position of all particles
    prev_pos    = pos;
    v           = F/xi;
    a = -2; b = 2; eta = (b-a).*rand(n,2) + a; % Brownian motion
    pos         = pos + v*dt + dt*eta;
    
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
