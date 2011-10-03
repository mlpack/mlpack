function [v, flag, theta, mu, data_diam, avg_data_sq_dist] = ...
    RPTree_split(data, P, c, dist_split_flag, dist_sq_mat)
% Input parameters:
% data: The data (X1, X2, ..., Xn)
% P: the set of the random unit (normal) vectors used for the
% random projections
% c: The parameter for the two different kinds of split on the
% basis of (data diameter) <= c (avg. sq. distances of data)
%
% Output parameters:
% v: The random unit direction chosen for this node if the split is
% done using random projection.
% flag: The flag which tells us whether the split was by distance (0)
% or by projection (1).
% theta: The (optimal) split point in the split by projection.
% mu: Split point in the split by distance. 
% data_diam: Diameter of the data.
% avg_data_sq_dist: Average squared distances between pairs of
% points in the data.


  [nDims, nPoints] = size(data);
  data_diam = 0;
  avg_data_sq_dist = 0;
  v = [];
  theta = 0;
  mu = 0;

  if nargin == 5
    % the pairwise distances are provided, easily compute shit (I
    % think)
    data_diam = max(max(dist_sq_mat));
    
    avg_data_sq_dist = sum(sum(dist_sq_mat)) / (nPoints * nPoints);
    
    
  else
    % Compute the diameter of the data and
    % the average squared distances of the data
    max_dist_list = [];
    temp_sum = 0;

    for i = 1:(nPoints -1)
      p = data(:, i);
      diffMat = data(:, (i+1: nPoints)) - ...
		repmat(p, [1 (nPoints - i)]);
      diffSqMat = diffMat .* diffMat;
      distSqVec = sum(diffSqMat, 1);
      max_dist_list(i) = max(distSqVec);
      temp_sum = temp_sum + sum(distSqVec);
      diffMat = [];
      diffSqMat = [];
      distSqVec = [];
    end
    
    data_diam = max(max_dist_list);
    avg_data_sq_dist = 2 * temp_sum / (nPoints * nPoints);
    
    %display(sprintf('diam.: %f, avg. diam.: %f', data_diam, ...
	%	    avg_data_sq_dist));
  end
  
  if data_diam <= c * avg_data_sq_dist
    % split by projection
    
    [d, numProjs] = size(P);
    
    best_proj = 0;
    best_proj_split = 0;
    best_proj_c = Inf;
    best_proj_split_size = 0;
    best_avg_diam = Inf;
    
    
    for i = 1:numProjs
      % trying every projection vector
      P_i = P(:, i);
      P_i_X = P_i' * data;
      [S I] = sort(P_i_X);
      
      best_c = Inf;
      best_split = 0;
      best_split_size = 0;
      
      % find the best split point
      for j = 1:(nPoints -1)
	mu_1 = mean(S(1:j));
	mu_2 = mean(S((j+1):nPoints));
	
	c = ((S(1:j) - mu_1) * (S(1:j) - mu_1)') + ...
	    ((S((j+1):nPoints) - mu_2) *...
	     (S((j+1):nPoints) - mu_2)');
	
	clear mu_1;
	clear mu_2;
	
	% finding the best split for this projection
	if c < best_c
	  best_c = c;
	  best_split = (S(j) + S(j+1)) / 2;
	  best_split_size = j;
	end
      end
      
      % finding the best projection with maximum decrease in
      % avg_diam
      % 
      % Not sure whether to use : FOR NOW
      %     avg_diam = 2 * \sum_x( (x - mean(X))^2 ) / |X|
      % or use : 
      %     avg_diam = 2 * \sum_x( (x - mean(X))^2 ) 
      %                      / (total # of points)
      
      
      % Compute avg_diam_left and avg_diam_right
      left_ind = find(P_i_X <= best_split); 
      right_ind = find(P_i_X > best_split);
      avg_diam_left = 0;
      avg_diam_right = 0;
      if nargin == 5
	numPoints = length(left_ind);
	tmp = dist_sq_mat(left_ind, :);
	dist_sq_mat_left = tmp(:, left_ind);
	
	avg_diam_left = sum(sum(dist_sq_mat_left)) / (numPoints * ...
						      numPoints);
	tmp = []; 
	dist_sq_mat_left = [];
	numPoints = length(right_ind);
	tmp = dist_sq_mat(right_ind, :);
	dist_sq_mat_right = tmp(:, right_ind);
	avg_diam_right = sum(sum(dist_sq_mat_right)) / (numPoints * ...
							numPoints);
      else
	leftData = data(:, left_ind);
	dMean = mean(leftData, 2);
	diffMat = leftData - repmat(dMean, [1 length(left_ind)]);
	diffSqMat = diffMat .* diffMat;
	distSqVec = sum(diffSqMat, 1);
	avg_diam_left = 2 * sum(distSqVec) / length(left_ind);
	
	dMean = [];
	diffMat = [];
	diffSqMat = [];
	distSqVec = [];
	
	rightData = data(:, right_ind);
	dMean = mean(rightData, 2);
	diffMat = rightData - repmat(dMean, [1 length(right_ind)]);
	diffSqMat = diffMat .* diffMat;
	distSqVec = sum(diffSqMat, 1);
	avg_diam_right = 2 * sum(distSqVec) / length(right_ind);
	
	dMean = [];
	diffMat = [];
	diffSqMat = [];
	distSqVec = [];
      end
      

      clear S;
      clear P_i_X;
      clear P_i;
      
      %display(sprintf('%d: %f -> %d/%d', i, best_c, ...
	%	      best_split_size, ...
	%             nPoints - best_split_size));

      
      avg_diam_child = avg_diam_left + avg_diam_right;
      if best_avg_diam > avg_diam_child
	% Previously: checking best projected avg_diam instead of
        % the best avg_diam in the real dataset.
	%
	% if best_c < best_proj_c
	best_avg_diam = avg_diam_child;
	best_proj_c = best_c;
	best_proj = i;
	best_proj_split = best_split;
	best_proj_split_size = best_split_size;
      end
      
    end
    
    v = P(:, best_proj);
    flag = 1;
    theta = best_proj_split;
      
    %display(sprintf('Best projection: %d, Split: %d/%d',...
	%	    best_proj, best_proj_split_size, ...
	%	    nPoints - best_proj_split_size));
    
  else
    % split by distance
    data_mean = mean(data, 2); 
    flag = 0;
    
    if dist_split_flag == 0
      % split by distance on the unprojected original data
      
      diffMat = data - repmat(data_mean, [1 nPoints]);
      diffSqMat = diffMat .* diffMat;
      distSqVec = sum(diffSqMat, 1);
    
      mu = median(distSqVec);
      
    else
      % split by distance on the randomly projected original data
      
      % project down for each RP and then compute the avg_diam for
      % each of the partition and use the one with the maximum
      % decrease
      [d, numProjs] = size(P);
	
      best_proj = 0;
      best_proj_split = 0;
      best_proj_mean = 0;
      best_proj_split_size = 0;
      best_avg_diam = Inf;
      
      for i = 1:numProjs
	% trying each projection vector
	P_i = P(:, i);
	P_i_X = P_i' * data;
	
	P_i_mean = P_i' * data_mean;
	
	split = median(abs(P_i_X - P_i_mean));
	
	
	% computing the avg_left_diam and the avg_right_diam
	left_ind = find(abs(P_i_X - P_i_mean) <= split);
	right_ind = find(abs(P_i_X - P_i_mean) > split);
	avg_diam_left = 0;
	avg_diam_right = 0;

	if nargin == 5
	  numPoints = length(left_ind);
	  tmp = dist_sq_mat(left_ind, :);
	  dist_sq_mat_left = tmp(:, left_ind);
	  
	  avg_diam_left = sum(sum(dist_sq_mat_left)) / (numPoints * ...
							numPoints);
	  tmp = []; 
	  dist_sq_mat_left = [];
	  numPoints = length(right_ind);
	  tmp = dist_sq_mat(right_ind, :);
	  dist_sq_mat_right = tmp(:, right_ind);
	  avg_diam_right = sum(sum(dist_sq_mat_right)) / (numPoints * ...
							  numPoints);
	else
	  
	  leftData = data(:, left_ind);
	  dMean = mean(leftData, 2);
	  diffMat = leftData - repmat(dMean, [1 length(left_ind)]);
	  diffSqMat = diffMat .* diffMat;
	  distSqVec = sum(diffSqMat, 1);
	  avg_diam_left = 2 * sum(distSqVec) / length(left_ind);
	  
	  
	  dMean = [];
	  diffMat = [];
	  diffSqMat = [];
	  distSqVec = [];
	  
	  rightData = data(:, right_ind);
	  dMean = mean(rightData, 2);
	  diffMat = rightData - repmat(dMean, [1 length(right_ind)]);
	  diffSqMat = diffMat .* diffMat;
	  distSqVec = sum(diffSqMat, 1);
	  avg_diam_right = 2 * sum(distSqVec) / length(right_ind);

	  dMean = [];
	  diffMat = [];
	  diffSqMat = [];
	  distSqVec = [];
	end	

	clear P_i_X;
	clear P_i;
	clear P_i_mean;

	avg_diam_child = avg_diam_left + avg_diam_right;
	
	
	% updating the current best projection
	if best_avg_diam > avg_diam_child
	  best_avg_diam = avg_diam_child;
	  best_proj = i;
	  best_proj_split = split;
	end
      end
      
      % choosing the best projection
      v = P(:, best_proj);
      mu = best_proj_split;
    end
  end
  