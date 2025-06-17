wipe

load('epigeom490sock_closed_aligned_shifted.mat')
% load('tank192_outw_struct.mat')
load('tank771_closed2_outw_struct.mat')
% load('selected_192_leads_frm_771.mat')
load('lungs_new_shifted.mat')
%   epigeom490corrected                1x1             38744  struct              
%   epigeom490sock_closed_aligned      1x1             35536  struct              
%   capped_sock_registered             1x1             34486  struct              
%   torso_tank                         1x1             55768  struct              
%   leads_from_771                   192x1              1536  double                                       
%   tank192                            1x1             13408  struct                                       


%% Plot the geometries

figure
% % 771-node torso
h=patch('Vertices',torso_tank.pts,'Faces',torso_tank.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'b';
h.FaceAlpha = 0;

% Closed heart geometry
h=patch('Vertices',epigeom490sock_closed_aligned.pts,'Faces',epigeom490sock_closed_aligned.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'r';
h.FaceAlpha = 0;
hold on

% Lung geometry
h=patch('Vertices',lungs_new.pts,'Faces',lungs_new.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'k';
h.FaceAlpha = 0.5;
hold off

%% Study the lungs geometry to separate right and left
lungs = lungs_new;

mean_x_plane = -12; %mean(lungs.pts(:,1)); Right/left lung ortasına denk gelen bir plane

max_y = max(lungs.pts(:,2));
min_y = min(lungs.pts(:,2));
max_z = max(lungs.pts(:,3));
min_z = min(lungs.pts(:,3));
plane.pts(1,:) = [mean_x_plane max_y max_z];
plane.pts(2,:) = [mean_x_plane min_y max_z];
plane.pts(3,:) = [mean_x_plane max_y min_z];
plane.pts(4,:) = [mean_x_plane min_y min_z];
plane.fac(1,:) = [1 3 4];
plane.fac(2,:) = [1 4 2];

% Plane'in nereye denk geldiğini kontrol
figure
h=patch('Vertices',lungs.pts,'Faces',lungs.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'b';
h.FaceAlpha = 0;
view(0,0)
% xlabel('x, mm')
% ylabel('y, mm')
% zlabel('z, mm')
hold on
% figure(2)
h=patch('Vertices',plane.pts,'Faces',plane.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'r';
h.FaceAlpha = 0.9;
h.FaceColor = 'r';
view(0,0)
xlabel('x, mm')
ylabel('y, mm')
zlabel('z, mm')
hold off

%% Fix the lung geometry - two sides separately!

% Right and left lung points were separated. 

find_x_sm = find(lungs.pts(:,1)<mean_x_plane);
right_lung.pts = lungs.pts(find_x_sm,:);

find_x_gt = find(lungs.pts(:,1)>mean_x_plane);
left_lung.pts = lungs.pts(find_x_gt,:);

% Plot the right and left lung points
h1 = plot3(right_lung.pts(:,1),right_lung.pts(:,2),right_lung.pts(:,3));
set(h1,'Marker','o','MarkerSize',5,'MarkerFaceColor','r','LineStyle','none');
hold on
% pause
h2 = plot3(left_lung.pts(:,1),left_lung.pts(:,2),left_lung.pts(:,3));
set(h2,'Marker','o','MarkerSize',5,'MarkerFaceColor','b','LineStyle','none');
view(0,0)
xlabel('x, mm')
ylabel('y, mm')
zlabel('z, mm')
hold off

% Find the indices of the coordinate rows in the whole lung geometry to
% grab node numbers
left_lung.nodes = find(ismember(lungs.pts,left_lung.pts,'rows'));
right_lung.nodes = find(ismember(lungs.pts,right_lung.pts,'rows'));

% Using node numbers, conditionally slice the overall FAC matrix
FAC = lungs.fac;
left_fac_indices = find(ismember(FAC(:,1),left_lung.nodes));
left_lung.fac = lungs.fac(left_fac_indices,:);
right_fac_indices = find(ismember(FAC(:,1),right_lung.nodes));
right_lung.fac = lungs.fac(right_fac_indices,:);

% The geometry triangularized together, when we split them the node numbers
% should be shifted to match the number of rows in pts matrix. Luckily, the
% the node numbers was in order, so the offset in the node numbers were
% subtracted from the fac matrix.

right_lung.fac = right_lung.fac - (min(min(right_lung.fac))-1) * ones(size(right_lung.fac));

% Lung geometry right - left 
h=patch('Vertices',right_lung.pts,'Faces',right_lung.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'b';
h.FaceColor = 'b';
h.FaceAlpha = 0.2;
hold on
h=patch('Vertices',left_lung.pts,'Faces',left_lung.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'r';
h.FaceColor = 'r';
h.FaceAlpha = 0.2;
hold off

%% Apply shift to the lungs and then plot

figure

% % 771-node torso
h=patch('Vertices',torso_tank.pts,'Faces',torso_tank.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'b';
h.FaceAlpha = 0;

% Closed heart geometry
h=patch('Vertices',epigeom490sock_closed_aligned.pts,'Faces',epigeom490sock_closed_aligned.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'r';
h.FaceAlpha = 0;
hold on

% % % Lung geometry
% % h=patch('Vertices',lungs_new.pts,'Faces',lungs_new.fac);
% % h.EdgeAlpha = 1;
% % h.EdgeColor = 'k';
% % h.FaceAlpha = 0.5;
% % hold off

% Plot shifted lungs
shift_amount_r = 1.175; % Right and left shift farklı olabilir. O nedenle iki ayrı variable var
right_lung_shifted = right_lung;
right_lung_shifted.pts(:,1) = right_lung_shifted.pts(:,1) - shift_amount_r;
h=patch('Vertices',right_lung_shifted.pts,'Faces',right_lung_shifted.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'b';
h.FaceAlpha = 0.5;

shift_amount_l = 0;
left_lung_shifted = left_lung;
left_lung_shifted.pts(:,1) = left_lung_shifted.pts(:,1) + shift_amount_l;
h=patch('Vertices',left_lung_shifted.pts,'Faces',left_lung_shifted.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'm';
h.FaceAlpha = 0.5;

%%
% Save the shifted lung geometries
lungs_shifted.vertices = [left_lung_shifted.pts;right_lung_shifted.pts];
lungs_shifted.faces = [left_lung_shifted.fac;right_lung_shifted.fac+size(left_lung_shifted.pts,1)];
heart.vertices = epigeom490sock_closed_aligned.pts;
heart.faces = epigeom490sock_closed_aligned.fac;


% Closed heart geometry
h=patch('Vertices',epigeom490sock_closed_aligned.pts,'Faces',epigeom490sock_closed_aligned.fac);
h.EdgeAlpha = 1;
h.EdgeColor = 'r';
h.FaceAlpha = 0;
hold on

% Shifted lung geometry
h=patch('Vertices',lungs_shifted.vertices,'Faces',lungs_shifted.faces);
h.EdgeAlpha = 1;
h.EdgeColor = 'k';
h.FaceAlpha = 0.5;
hold off

%%%%% Buraya çakışıp çakışmadığını kontrol eden fonksiyonu koyabilirsin.

[intersection,surface] = SurfaceIntersection(heart,lungs_shifted);

display(['Number of intersections: ',num2str(sum(intersection,'all'))])
%%%%% Çakışmayan lung geometrisini save edip kullanırsın.

save('lungs_shifted',"lungs_shifted")
