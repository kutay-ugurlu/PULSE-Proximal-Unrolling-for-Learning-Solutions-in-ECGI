clear
L = load('rsm10jan01-cs-0014_feat.mat');
potvals = L.ts.potvals;
real_AT = L.features.AT;

QRSBegin = 160;
QRSEnd = 250;

potvals = potvals(:,QRSBegin:QRSEnd);
real_AT = real_AT - QRSBegin;
geom_file = load("epigeom490corrected.mat");
geom = geom_file.epigeom490corrected;
% ind = find(AT_min>600);
SPCoh_AT = SpCoherentActTime(potvals, geom);