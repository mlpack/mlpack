function [v, flag, theta, mu, data_diam, avg_data_sq_dist] = ...
    RPTree_split2(data, P, c)
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

 
  % split by projection
  [d, numProjs] = size(P);
    
  best_proj = 0;
  best_proj_split = 0;
  best_proj_c = Inf;
  best_proj_split_size = 0;
  best_avg_diam = Inf;
  
  best_data_diam = 0;
  best_avg_data_sq_dist = 0;
      
  for proj_i = 1:numProjs
    % trying every projection vector
    P_i = P(:, proj_i);
    P_i_X = P_i' * data;
    [S I] = sort(P_i_X);
    
    data_diam = (S(nPoints) - S(1))^2;
    temp_sum = 0;

    for i = 1:(nPoints -1)
      p = S(i);
      diffVec = S((i+1: nPoints)) - ...
		repmat(p, [1 (nPoints - i)]);
      diffSqVec = diffVec .* diffVec;
      temp_sum = temp_sum + sum(diffSqVec);
      diffVec = [];
      diffSqVec = [];
    end
    avg_data_sq_dist = 2 * temp_sum / (nPoints * nPoints);
    
    %display(sprintf('diam.: %f, avg. diam.: %f', data_diam, ...
	%	    avg_data_sq_dist));
    %end

  
    if data_diam <= c * avg_data_sq_dist
      
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
      
      P_i_X_l = P_i_X(left_ind);
      numPoints = length(left_ind);
      for i = 1:(numPoints -1)
	p = P_i_X_l(i);
	diffVec = P_i_X_l((i+1: numPoints)) - ...
		  repmat(p, [1 (numPoints - i)]);
	diffSqVec = diffVec .* diffVec;
	avg_diam_left = avg_diam_left + sum(diffSqVec);
	diffVec = [];
	diffSqVec = [];
      end
      avg_diam_left = 2 * avg_diam_left / (numPoints * numPoints);

      P_i_X_r = P_i_X(right_ind);
      numPoints = length(right_ind);
      for i = 1:(numPoints -1)
	p = P_i_X_r(i);
	diffVec = P_i_X_r((i+1: numPoints)) - ...
		  repmat(p, [1 (numPoints - i)]);
	diffSqVec = diffVec .* diffVec;
	avg_diam_right = avg_diam_right + sum(diffSqVec);
	diffVec = [];
	diffSqVec = [];
      end
      avg_diam_right = 2 * avg_diam_right / (numPoints * numPoints);


      
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
	best_proj = proj_i;
	best_proj_split = best_split;
	best_proj_split_size = best_split_size;
	flag = 1;
	best_data_diam = data_diam;
	best_avg_data_sq_dist = avg_data_sq_dist;
      end
      
      clear S;
      clear P_i_X;
      clear P_i;
      %end
      %display(sprintf('Best projection: %d, Split: %d/%d',...
      %	    best_proj, best_proj_split_size, ...
      %	    nPoints - best_proj_split_size));
    
    else
      % split by distance
      data_mean = mean(data, 2); 
      %flag = 0;
    
      % split by distance on the randomly projected original data
      
      % project down for each RP and then compute the avg_diam for
      % each of the partition and use the one with the maximum
      % decrease
	
      P_i_mean = P_i' * data_mean;
	
      split = median(abs(P_i_X - P_i_mean));
	
	
      % computing the avg_left_diam and the avg_right_diam
      left_ind = find(abs(P_i_X - P_i_mean) <= split);
      right_ind = find(abs(P_i_X - P_i_mean) > split);
      avg_diam_left = 0;
      avg_diam_right = 0;

      P_i_X_l = P_i_X(left_ind);
      numPoints = length(left_ind);
      for i = 1:(numPoints -1)
	p = P_i_X_l(i);
	diffVec = P_i_X_l((i+1: numPoints)) - ...
		  repmat(p, [1 (numPoints - i)]);
	diffSqVec = diffVec .* diffVec;
	avg_diam_left = avg_diam_left + sum(diffSqVec);
	diffVec = [];
	diffSqVec = [];
      end
      avg_diam_left = 2 * avg_diam_left / (numPoints * numPoints);

      P_i_X_r = P_i_X(right_ind);
      numPoints = length(right_ind);
      for i = 1:(numPoints -1)
	p = P_i_X_r(i);
	diffVec = P_i_X_r((i+1: numPoints)) - ...
		  repmat(p, [1 (numPoints - i)]);
	diffSqVec = diffVec .* diffVec;
	avg_diam_right = avg_diam_right + sum(diffSqVec);
	diffVec = [];
	diffSqVec = [];
      end
      avg_diam_right = 2 * avg_diam_right / (numPoints * numPoints);


      avg_diam_child = avg_diam_left + avg_diam_right;
      % updating the current best projection
      if best_avg_diam > avg_diam_child
	best_avg_diam = avg_diam_child;
	best_proj = proj_i;
	best_proj_split = split;
	flag = 0;
	best_data_diam = data_diam;
	best_avg_data_sq_dist = avg_data_sq_dist;
      end

      clear P_i_X;
      clear P_i;
      clear P_i_mean;
      
    end
  end
  

  v = P(:, best_proj);
  if flag == 1;
    theta = best_proj_split;
  else
    mu = best_proj_split;
  end

  data_diam = best_data_diam;
  avg_data_sq_dist = best_avg_data_sq_dist;