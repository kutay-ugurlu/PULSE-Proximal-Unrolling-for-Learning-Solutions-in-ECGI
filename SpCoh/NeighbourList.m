function [ neighbours ] = NeighbourList(triangles)

% neighbourList   Find the 1st order neighbours
%
% Usage: 
%        neighbours = neighbourList(triangles);
%
% Inputs:
%   triangles   geom.fac (size: (3 x numOfTri); 
%               node indexing should start with 1, not 0)
%
% Output:
%   neighbours  (1st order) neighours matrix:
%                1st column: lead index, 
%                2nd column: number of neighbours
%                3rd-last column: lead indices of neighbours
% 
%
% Author: Onder Nazim Onak ononak@gmail.com
%
if(size(triangles,1) ~= 3)
    error('Invalid triangle data')
end

 vertice_numbers = unique(triangles)';

 maxneighbour = 0;
 for vtxId = vertice_numbers
     
     [rowInd colInd] = find(triangles == vtxId);
     tmp = unique(reshape(triangles(:,colInd),1,[]));
     neighList{vtxId} = tmp(tmp ~= vtxId);
     
     if(length(neighList{vtxId}) > maxneighbour)
         maxneighbour = length(neighList{vtxId});
     end
 end

 neighbours = zeros(max(vertice_numbers), maxneighbour + 2);
 
 for vtxId = vertice_numbers
     neighbours(vtxId,1) = vtxId;
     neighbours(vtxId,2) = length(neighList{vtxId});
     neighbours(vtxId,3:2+length(neighList{vtxId})) = neighList{vtxId};
 end
 

end

