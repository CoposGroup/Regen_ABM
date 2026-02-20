% Agent-based stochastic model
% Velocity Test
% Epidermis is loaded from CTRL T2
%
% SW (JUN 2024)


close all;
% clc;
clear;


% Load data
load('CTRL_data.mat', ...
     'v_2_4_CTRL_m', ...
     'v_4_6_CTRL_m', ...
     'v_6_8_CTRL_m', ...
     'v_8_10_CTRL_m', ...
     'v_10_12_CTRL_m', ...
     'v_2_12_CTRL_m', ...
     'CTRL_T2_I', ...
     'v_2_6_CTRL_m', ...
     'CTRL_T12_shifted_I');


% Video
video_flag = 0;
if(video_flag)
    vidObj = VideoWriter('test.mp4','MPEG-4');
    open(vidObj);
end


% Parameters
dt          = 0.001;
Tmax        = 5.0;


Xb_CTRL_T2 = CTRL_T2_I;
Xb_CTRL_T2_og = Xb_CTRL_T2;


% noisy velocity
vx_new = v_2_12_CTRL_m(:,1) + 0.1*mean(v_2_12_CTRL_m(:,1))*rand(length(v_2_12_CTRL_m),1);
vy_new = v_2_12_CTRL_m(:,2) + 0.1*mean(v_2_12_CTRL_m(:,2))*rand(length(v_2_12_CTRL_m),1);
v_2_12_CTRL_m_noisy = [vx_new,vy_new];



for t = 1:Tmax/dt
    % update position
    Xb_CTRL_T2 = Xb_CTRL_T2 + dt*(v_2_12_CTRL_m_noisy);

    if min(Xb_CTRL_T2(:,2)) <= min(CTRL_T12_shifted_I(:,2))
        break;
    end

    

    if mod(t,100) == 0
        % plotting
        figure(1);
        plot(Xb_CTRL_T2(:,1), Xb_CTRL_T2(:,2), 'k','linewidth',2); hold on;

        plot(CTRL_T12_shifted_I(:,1),CTRL_T12_shifted_I(:,2),'g','LineWidth',2)
        plot(Xb_CTRL_T2_og(:,1),Xb_CTRL_T2_og(:,2),'r','LineWidth',2);

        % quiver(Xb_rotated(:,1),Xb_rotated(:,2),v_2_12_CTRL_m(:,1),v_2_12_CTRL_m(:,2),'-r');
        xlim([-700 -250]); ylim([580 760]); box on; grid off;
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


save('Xb_CTRL_T2_final.mat','Xb_CTRL_T2')


function shape = interpolate(x, y, N)
    x_new = linspace(min(x),max(x),N);
    y_new = interp1(x,y,x_new);

    x_new_T = transpose(x_new);
    y_new_T = transpose(y_new);

    shape = [x_new_T,y_new_T];
end

