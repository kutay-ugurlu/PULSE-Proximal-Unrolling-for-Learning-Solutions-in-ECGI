function [rotated,M,rotation_str] = rotate_by(pts,direction,angle)

[s1,~] = size(pts);
if s1 == 3
    pts = pts';
end

%% Find the shift vector from the origin 
shift = mean(pts,1);
pts = pts-shift;

switch direction 
    case 'x'
        M = [1 0 0; 0 cosd(angle) -sind(angle); 0 sind(angle) cosd(angle)];
    case 'y'
        M = [cosd(angle) 0 -sind(angle); 0 1 0; sind(angle) 0 cosd(angle)];
    case 'z'
        M = [cosd(angle) -sind(angle) 0; sind(angle) cosd(angle) 0; 0 0 1];
end
    rotated = (M*pts' + shift')' ;
    rotation_str = [direction,'_by_',num2str(angle)]; 
end