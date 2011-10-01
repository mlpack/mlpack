function [T] = RPTree_build2(data, P, c, indices)

% Using RP splits the tree is built hierarchically.
% 
% 'data':  Should be column major, with every column 
% denoting a point.
%
% P: the dictionary of random projections to use while building
% the tree.
%
% c: The parameter for deciding whether to choose split by random
% projection or split by distance.
% 
% indices: The indices of the data passed on to this node
% 
% dist_sq_mat: The matrix containing the pairwise squared distances
% of all the points passed to this node.

if nargin == 3
  indices = 1:size(data,2);
end



T.indices = indices;
T.leaf = 0;
T.left = [];
T.right = [];
T.flag = 0;
T.mu = 0;
T.data_mean = [];
T.v = [];
T.theta = 0;
T.time = 0;
T.left_limit = 0;
T.right_limit = 0;
T.dist_split_flag = 1;
T.leaf_sum = 0;

% if the data size is small, treat it as a leaf
% if size(data, 2) < 30
if length(indices) <= 30
  T.leaf = 1;
  nPoints = length(indices);
  
  % computing the data_diam and avg_diam
  max_dist_list = [];
  temp_sum = 0;
    
  nodeData = data(:, indices);
  for i = 1:(nPoints -1)
    p = nodeData(:, i);
    diffMat = nodeData(:, (i+1: nPoints)) - ...
	      repmat(p, [1 (nPoints - i)]);
    diffSqMat = diffMat .* diffMat;
    distSqVec = sum(diffSqMat, 1);
    max_dist_list(i) = max(distSqVec);
    temp_sum = temp_sum + sum(distSqVec);
    diffMat = [];
    diffSqMat = [];
    distSqVec = [];
  end
    
  T.data_diam = max(max_dist_list);
  T.avg_diam = 2 * temp_sum / (nPoints * nPoints);
  %display(sprintf('diam.: %f, avg. diam.: %f', T.data_diam, ...
  %	  T.avg_diam));
  
  T.leaf_sum = T.avg_diam * nPoints;

% Step 1: Find the best split
% Step 2: Partition the data using the split
% Step 3: Recurse on each partition
else
  nodeData = data(:, indices);
   
  tic;
  v = [];
  theta = [];
  mu = [];
  
  [v, T.flag, theta, mu, T.data_diam, T.avg_diam] = ...
      RPTree_split2(nodeData, P, c);
    
  T.time = toc;

  left_indices = [];
  right_indices = [];
  left_ind = [];
  right_ind = [];
  
  if T.flag == 1
    % split by projection
    T.v = v;
    T.theta = theta;
    
    P_i_X = v' * nodeData;
    
    left_ind = find(P_i_X <= theta);
    T.left_limit = max(P_i_X(left_ind));
    right_ind = find(P_i_X > theta);
    T.right_limit = min(P_i_X(right_ind));
    
    left_indices = indices(left_ind);
    right_indices = indices(right_ind);
    
  else
    % split by distance
    T.mu = mu;
    T.data_mean = mean(nodeData, 2);
    left_ind = [];
    right_ind = [];
    
    % split by distance on the randomly projected data
    T.v = v; 
      
    proj_mean = T.v' * T.data_mean;
    P_i_X = T.v' * nodeData;
    
    left_ind = find ( abs(P_i_X - proj_mean) <= T.mu);
    T.left_limit = max( abs(P_i_X(left_ind) - proj_mean) );
    right_ind = find ( abs(P_i_X - proj_mean) > T.mu);
    T.right_limit = min( abs(P_i_X(right_ind) - proj_mean) );
        
    left_indices = indices(left_ind);
    right_indices = indices(right_ind);
  end
  
  %display(sprintf('%d:%d - %d/%d', length(indices), T.flag, ...
  %	  length(left_indices), length(right_indices)));
  
  T.left = RPTree_build2(data, P, c, left_indices);
  T.right = RPTree_build2(data, P, c, right_indices);
  
  T.leaf_sum = T.left.leaf_sum + T.right.leaf_sum;
end
