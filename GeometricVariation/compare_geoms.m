reordering = load("newnode_order_3.mat").node_order;
load('epigeom490corrected.mat');
load('epigeom490sock_closed_aligned_shifted.mat') % heart
pts1 = epigeom490sock_closed_aligned.pts;
pts2 = epigeom490corrected.pts;
x1 = pts1(:,1);
y1 = pts1(:,2);
z1 = pts1(:,3);
x2 = pts2(:,1);
y2 = pts2(:,2);
z2 = pts2(:,3);


scatter3(x1,y1,z1)