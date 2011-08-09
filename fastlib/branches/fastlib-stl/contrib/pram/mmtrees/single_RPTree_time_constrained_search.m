% The following function performs the recursion on the tree while
% searching for the nearest neighbor.
function [q] = single_RPTree_time_constrained_search(q, R, T)

% either we are computing exact or the required number of leaves
% have not been visited
%if (q.compute_exact == 1 | q.ann_check > 0)

    % if leaf, do naive search
    if T.leaf == 1
      
      nPoints = size(T.indices, 2); 
      nSamples = 0;
      
      
      %[nDim nPoints] = size(leafData);
      
      %display(sprintf('Leaf: %d points', nPoints));
      % if leaf big, then just sample
      leafData = [];
      %samplesIndices = [];
      %if (q.compute_exact == 0) & (ceil(nPoints * beta) > 20)
	%nSamples = ceil(nPoints * beta);
	%sampleIndices = randi([1 nPoints], nSamples, 1);
	%leafData = R(:, T.indices(sampleIndices));
      %else
	nSamples = nPoints;
	sampleIndices = [1:nPoints];
	leafData = R(:, T.indices);
      %end      

      diffMat = leafData - repmat(q.p, [1 nSamples]);
      diffSqMat = diffMat .* diffMat;
      
      distSqVec = sum(diffSqMat, 1);
      
      [dist ind] = min(distSqVec);
      
      q.nn_dist_comp = q.nn_dist_comp + nSamples;
      
      if dist < q.ub
	q.ub = dist;
	q.nn = T.indices(sampleIndices(ind));
	
	%if q.ann_check > 0
	%q.ann = T.indices(ind);
	%q.ann_dist = dist;
	%end
      end
      
      %if q.compute_exact == 0
	%q.samples_made = q.samples_made + nSamples;
	%q.points_seen = q.points_seen + nPoints;
	%q.from_leaf = q.from_leaf + nSamples;
	
	q.num_leaves = q.num_leaves + 1;
	rank_error = q.rank_vec(q.nn) - 1;
	q.rank_error_list(q.num_leaves) = rank_error;
	q.dist_comp_list(q.num_leaves) = q.nn_dist_comp;

      %end
      %if q.ann_check == 0
      %  q.ann_check = q.ann_check - 1;
      %end
    else
      %recurse down the reference tree as usual     
      % Tighter bounds on the hyperplanes are added
      %
      % This provides you with tighter
      % upperbounds on the possible nn distance.

      q.nn_marg_comp = q.nn_marg_comp + 1;
	  
      % perform recursion
      if T.flag == 1
	% split by projection
	    
	class = T.v' * q.p - T.theta;
	    
	if class <= 0
	  % go down the left subtree
	  q = single_RPTree_time_constrained_search(q, R, T.left);
	  	      
	  % perform backtracking
	  dSqToMargin = (T.right_limit - T.theta - class) * ...
	      (T.right_limit - T.theta - class) /...
	      sum(T.v .* T.v);
	      
	  if dSqToMargin < q.ub
	    q = single_RPTree_time_constrained_search(q, R, ...
						      T.right);
	    
	  %else
	  %  q.points_seen = q.points_seen +...
	  %  size(T.right.indices, 2);
	  %  q.samples_made = q.samples_made +...
	  %  floor(beta * size(T.right.indices, 2));
	  end
	      
	else
	  % go down the right subtree
	  q = single_RPTree_time_constrained_search(q, R, T.right);
	  
	  % perform backtracking
	  dSqToMargin = (class + T.theta - T.left_limit) *...
	      (class + T.theta - T.left_limit) / ...
	      sum(T.v .* T.v);
	      	      
	  if dSqToMargin < q.ub
	    q = single_RPTree_time_constrained_search(q, R, ...
						      T.left);
	    
	    %  else
	    %q.points_seen = q.points_seen +...
	    %    size(T.left.indices, 2);
	    %q.samples_made = q.samples_made +...
	    %    floor(beta * size(T.left.indices, 2));
	  end
	end
	    	    
      else
	% split by distance
	    
	diffVec = q.p - T.data_mean;
	distSq = sum (diffVec .* diffVec);
	    
	if T.dist_split_flag == 0
	  % dealing with the real points and distances, not the
          % projected version of the points
	  if distSq <= T.mu
	    % go down the left subtree
	    q = single_RPTree_time_constrained_search(q, R, ...
						      T.left);
	    %check for backtrack
	    if distSq^0.5 + q.ub^0.5 >= T.right_limit^0.5
	      %backtrack
	      q = single_RPTree_time_constrained_search(q, R, ...
							T.right);
	      
	      % else
	      %  q.points_seen = q.points_seen +...
	      %  size(T.right.indices, 2);
	      %  q.samples_made = q.samples_made +...
	      %  floor(beta * size(T.right.indices, 2));
	    end
	  else
	    % go down the right subtree
	    q = single_RPTree_time_constrained_search(q, R, ...
						      T.right);
	    	
	    %check for backtrack
	    if distSq^0.5 - q.ub^0.5 <= T.left_limit^0.5
	      %backtrack
	      q = single_RPTree_time_constrained_search(q, R, ...
							T.left);
	      % else
	      %  q.points_seen = q.points_seen +...
	      %      size(T.left.indices, 2);
	      %  q.samples_made = q.samples_made +...
	      %      floor(beta * size(T.left.indices, 2));
	    end
	  end
	else
	  % operate on the projected data
	      
	  absDiff = abs(T.v' * diffVec);
	  if absDiff <= T.mu
	    % go down the left subtree
	    q = single_RPTree_time_constrained_search(q, R, ...
						      T.left);
	    	
	    %check for backtrack
	    %proj_q_ub = absDiff * q.ub^0.5 / distSq^0.5;
	    if absDiff + q.ub^0.5 >= T.right_limit
	      %backtrack down the right subtree
	      q = single_RPTree_time_constrained_search(q, R, ...
							T.right);
	      
	      % else
	      % prune the right subtree
	      %  q.points_seen = q.points_seen +...
	      %      size(T.right.indices, 2);
	      %  q.samples_made = q.samples_made +...
	      %      floor(beta * size(T.right.indices, 2));
	    end
	  else
	    %go down the right subtree
	    q = single_RPTree_time_constrained_search(q, R, ...
						      T.right);
	    		
	    %check for backtrack
	    %proj_q_ub = absDiff * q.ub^0.5 / distSq^0.5;
		
	    if absDiff - q.ub^0.5 <= T.left_limit
	      % backtrack down the left subtree
	      q = single_RPTree_time_constrained_search(q, R, ...
							T.left);
	      		  
	      % else
	      % prune the left subtree
	      %  q.points_seen = q.points_seen +...
	      %      size(T.left.indices, 2);
	      %  q.samples_made = q.samples_made +...
	      %      floor(beta * size(T.left.indices, 2));
	    end % backtrack
	  end % projected data
	end % split by distance
      end
    end % if leaf
    %  end
  %  end
%  end
%end
 
