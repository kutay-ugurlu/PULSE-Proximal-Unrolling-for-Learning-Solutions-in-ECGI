function [ lat] = LocalActTime(signal)
%LOCALACTTIME Computes the local activation time
%
% Author: Onder Nazim Onak ononak@gmail.com

 [nlead,~] = size(signal);
 lat = zeros(nlead,1);

 [filteredSignal] = SmoothingFilter(signal, 3,31);
 
 for i=1:nlead       
  %[dif difHR fsignal] = DiffByRegularization(signal(i,:));
   dif = gradient(filteredSignal(i,:));
   [~, ind] = min(dif);
   lat(i,1) = ind;
   %h = ntime/(length(difHR(1,:)) - 1); 
%       [~, ind] = min(difHR);
%       [lat(i,1)] = ind*h;
 end

end


