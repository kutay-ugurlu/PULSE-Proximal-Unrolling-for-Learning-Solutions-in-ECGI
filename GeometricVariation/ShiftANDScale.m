function mysurface = ShiftANDScale(mysurface, shiftVec, myScale)

% Assumes pts is in 3xN format!!

% Apply scale
meanpts = repmat(mean(mysurface.pts,2),1, size(mysurface.pts,2));
mysurface.pts = mysurface.pts - meanpts;
scaledpts = mysurface.pts*myScale + meanpts; % eski koordinatlara tasi

% Apply shift
shiftedpts = scaledpts + shiftVec;
mysurface.pts = shiftedpts;
end
