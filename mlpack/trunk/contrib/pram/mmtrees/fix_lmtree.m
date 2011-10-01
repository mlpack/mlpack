function [] = fix_lmtree(T, R)
% This function goes through a built large margin tree and attempts
% to understand why certain nodes were unable to find large margin
% splits
%
% Input:
% T: The large margin tree
% R: The reference data set
%
% Output:

if T.leaf == 1
  if length(T.indices) > 30
    % this is where we were unable to split and gave up
    
    % compute max diam
    % compute avg diam
    [max_diam, avg_diam, dist_set] = diam_stats(R(:, T.indices));
    c = max_diam / avg_diam;

    display(sprintf('***|L| = %d, c_norm = %0.2f', ...
		    length(T.indices), c / length(T.indices)));
    %keyboard;
    num_dist = length(dist_set);
    [N, X] = hist(dist_set, 10);
    plot(X, N / num_dist, '-r');
  end
else
  % compute max_diam, avg_diam, c
  [max_diam, avg_diam, dist_set] = diam_stats(R(:, T.indices));
  c = max_diam / avg_diam;
  
  display(sprintf('c_norm = %0.2f, %d:%d/%d', c / length(T.indices),...
		  length(T.indices), ...
		  length(T.left.indices), ...
		  length(T.right.indices)));
  %keyboard;
  
  num_dist = length(dist_set);
  [N, X] = hist(dist_set, 10);
  if length(T.indices) < 60
    plot(X, N / num_dist, '-.k');
  end
  % continue recursing
  fix_lmtree(T.left, R);
  fix_lmtree(T.right, R);
end

% end fix_lmtree

function [max_diam, avg_diam, dist_set] = diam_stats(S)
% Computes the maximum diameter and the average pairwise distances
% for the given data set
[nDims, nPoints] = size(S);

max_dist_list = [];
temp_sum = 0;
dist_set = [];

for i = 1:(nPoints -1)
  p = S(:, i);
  diffMat = S(:, (i+1: nPoints)) - ...
	    repmat(p, [1 (nPoints - i)]);
  diffSqMat = diffMat .* diffMat;
  distSqVec = sum(diffSqMat, 1);
  max_dist_list(i) = max(distSqVec);
  temp_sum = temp_sum + sum(distSqVec);
  dist_set = [dist_set, distSqVec];
  diffMat = [];
  diffSqMat = [];
  distSqVec = [];
end

max_diam = max(max_dist_list);
avg_diam = 2 * temp_sum / (nPoints * nPoints);

