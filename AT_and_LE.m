function [AT_CC,Localization_Error,AT,estimated_pace_loc] = AT_and_LE(Xinv, Real_PaceLoc, real_AT, test_bads)
% Bayesian Solution and Saving metrics coded in a script to avoid copy
% paste 
geom = load('reordered_epigeom490corrected.mat').reordered_geom;
addpath('SpCoh') 

% PTS = geom.pts;
% L = surface_laplacian(geom);
% % Now, eliminate bad_rows
% test_valids = setdiff(1:size(Xinv,1),test_bads);
% % Xinv = Xinv(test_valids,:);
% % X_test = X_test(test_valids,:);

% % Compute Real AT here
% AT = ActivationTimeST(Xinv,L);
% % AT = ActivationTime(Xinv);
% AT_CC = calculate_cc(real_AT(test_valids),AT(test_valids));
% estimated_pace_loc = find(AT == min(AT),1);
% Localization_Error = vecnorm(PTS(Real_PaceLoc,:)-PTS(estimated_pace_loc,:),2,2);


PTS = geom.pts;
% Now, eliminate bad_rows
test_valids = setdiff(1:size(Xinv,1),test_bads);
% Xinv = Xinv(test_valids,:);
% X_test = X_test(test_valids,:);

% Compute Real AT here
AT = SpCoherentActTime(Xinv, geom);
AT_CC = calculate_cc(real_AT(test_valids),AT(test_valids));
AT_copy = AT;
AT_copy(test_bads) = nan;
[~,estimated_pace_loc] = min(AT_copy);
Localization_Error = vecnorm(PTS(Real_PaceLoc,:)-PTS(estimated_pace_loc,:),2,2);
estimated_pace_loc = estimated_pace_loc - 1;
end