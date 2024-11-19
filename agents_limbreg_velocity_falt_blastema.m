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


% Load velocity data
load('CTRL_Velocity.mat','v_2_12_CTRL_m');



x = linspace(-5,5,1000);
y = -1*ones(size(x));

flat_blastema = [x(:),y(:)];
flat_blastema_og = [x(:),y(:)];

% plot(x,y)






for t = 1:Tmax/dt

    flat_blastema = flat_blastema + dt*(v_2_12_CTRL_m);


    if mod(t,100) == 0
        % plotting
        figure(1);
        plot(flat_blastema(:,1), flat_blastema(:,2), 'k','linewidth',2);

        hold on;

        plot(flat_blastema_og(:,1),flat_blastema_og(:,2),'r','LineWidth',2);

        % quiver(Xb_rotated(:,1),Xb_rotated(:,2),v_2_12_CTRL_m(:,1),v_2_12_CTRL_m(:,2),'-r');
        xlim([-20 20]); ylim([-12 0.5]); box on; grid off;
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



if(video_flag)
    close(vidObj);
end


