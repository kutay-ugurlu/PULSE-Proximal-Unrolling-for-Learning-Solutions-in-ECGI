function [ filteredSignal ] = SmoothingFilter(signal, polyOrder, frameLenght)
%SMOOTHINGFILTER use polynomial smoothing filter for denoising
%
% Author: Onder Nazim Onak ononak@gmail.com
%
filteredSignal = zeros(size(signal));

if(nargin == 2)
    order = polyOrder;
    framelen = 11;
elseif(nargin == 3)
    order = polyOrder;
    framelen = frameLenght;     
else
    order = 3;
    framelen = 11;
end
        
    for i=1:size(signal,1)
        filteredSignal(i,:) = sgolayfilt(signal(i,:),order,framelen);
    end
end

