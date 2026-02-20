clear
close all
% clc

%% From Shawn's file
Nb = 50;
Ncells = 200;

% Form semicircle
semi_circle_center = [0, 0];
semi_circle_radius = 1.5;
theta = linspace(3*pi/2,5*pi/2,Nb);
xb_semi_circle = semi_circle_radius * cos(theta) + semi_circle_center(1);
yb_semi_circle = semi_circle_radius * sin(theta) + semi_circle_center(2);

% Define the vertical line segments
x_vert = [1.2, 1.2];

% Determine y-values on the semi-circle
diffs = abs(xb_semi_circle - x_vert(1));
[~, sorted_indices] = sort(diffs);
closest_indices = sorted_indices(1:2); % find the indices of the two smallest differences
val_yb = yb_semi_circle(closest_indices);
val_xb = xb_semi_circle(closest_indices);
y_vert = val_yb;

% Find mean spacing
xb_space = xb_semi_circle(1:min(closest_indices));
yb_space = yb_semi_circle(1:min(closest_indices));
for i=1:length(xb_space)-1
    ds = sqrt((xb_space(i+1)-xb_space(i))^2+(yb_space(i+1)-yb_space(i))^2);
end
avg_ds = mean(ds);

% Set "cut" (vertical line) segment
y_v = y_vert(2):avg_ds:y_vert(1);
x_v = val_xb(1)*ones(1,length(y_v));

% Combine the semicircle and the vertical line segment in the correct order
xb = [xb_semi_circle(1:min(closest_indices)),x_v,xb_semi_circle(max(closest_indices):end)];
yb = [yb_semi_circle(1:min(closest_indices)),y_v,yb_semi_circle(max(closest_indices):end)];

% Define the epidermis boundary
Xb0 = [xb;yb]'; 
Xb = [xb;yb]';
dsb = sqrt( (Xb0(1,1)-Xb0(2,1))^2 + (Xb0(1,2)-Xb0(2,2))^2 ); % initial distance btw the first 2 pts on the b

% Set the interior mesh; cells are placed at node locations
inds = find(Xb(:,1)<0.00001);
y2 = Xb(inds(2),2);
y1 = Xb(inds(1),2);
yy = y1:mean(dsb):y2;
yy = yy';
xx = zeros(length(yy),1);
Xb = [Xb;[xx yy]];
pg = polyshape([Xb]);
tr = triangulation(pg);
gm = fegeometry(tr);
pdegplot(gm)
hold on;
gm = generateMesh(gm);
pos0 = gm.Mesh.Nodes'; 

% Rescale points to ensure inside the domain
pos0 = 0.98*pos0;

% Plot points and sub-sample
scatter(pos0(:,1),pos0(:,2),'ro');
idsample = randi(length(pos0),Ncells,1);
scatter(pos0(idsample,1),pos0(idsample,2),'k*');

pos0 = pos0(idsample,:);

% Save cell locations
initfile = 'cellinitialization_n200.mat';
save(initfile,'pos0','Ncells');


