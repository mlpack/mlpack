function [omega, b, time] = KDTree_median_split(data)
% do a kdtree median split on the dimension with the highest range
% OPTIONS:  either do it recursively or just return the split
%
% FOR NOW: return the split

tic;
[nDim, nPoints] = size(data);

rangeData = max(data, [], 2) - min(data, [], 2);
[maxRange, splitInd] = max(rangeData);
%display(splitInd);

splitPoint = median(data(splitInd, :));

%display(splitPoint);

omega = zeros(nDim, 1);
omega(splitInd) = 1.0;
if splitPoint == min(data(splitInd, :))
  b = - (splitPoint + (0.0001 * maxRange));
else
  b = -splitPoint;
end

time = toc;
