%% reorder the epigeom geometry
wipe
reordering = load("newnode_order_3.mat").node_order;
geom = load('epigeom490sock_closed_aligned_shifted.mat').epigeom490sock_closed_aligned;
old_pts = geom.pts;
old_fac = geom.fac;

new_pts = old_pts(reordering,:);
new_fac = arrayfun(@(x) find(x==reordering),old_fac);

trisurf(new_fac,new_pts(:,1),new_pts(:,2),new_pts(:,3))

epigeom490sock_closed_aligned = struct();
epigeom490sock_closed_aligned.pts = new_pts;
epigeom490sock_closed_aligned.fac = new_fac;
save("reordered_epigeom490sock_closed_aligned_shifted.mat","epigeom490sock_closed_aligned");

