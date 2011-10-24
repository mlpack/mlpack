% The following function performs the recursion on the tree while
% searching for the nearest neighbor.
function [q] = single_LMTree_time_constrained_search(q, R, T)

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
	
	% collecting accuracy obtained at this leaf level
	q.num_leaves = q.num_leaves + 1;
	rank_error = q.rank_vec(q.nn) - 1;
	q.rank_error_list(q.num_leaves) = rank_error;
	q.dist_comp_list(q.num_leaves) = q.nn_dist_comp;
	
	%display(sprintf('%d->%d, %d',rank_error, q.num_leaves, q.nn));
      %end
      %if q.ann_check == 0
      %  q.ann_check = q.ann_check - 1;
      %end
    else
      
      %recurse down the reference tree as usual     
      % TWEAK
      % Add information in the tree about the min(abs(omega'*x + b))
      % for each of the children and that provides you with tighter
      % upperbounds on the possible nn distance.

      
      % perform recursion
      class = T.omega' * q.p + T.b;
      q.nn_marg_comp = q.nn_marg_comp + 1;

      if class < 0
	% go down the left subtree
	%display(sprintf('Going Left'));
	q = single_LMTree_time_constrained_search(q, R, T.left);
	
	% perform backtracking
	dSqToMargin = (T.right_min - class) *...
	    (T.right_min - class)...
	    / sum(T.omega .* T.omega);

	if dSqToMargin < q.ub
	  %display(sprintf('Left backtrack...'));
	  q = single_LMTree_time_constrained_search(q, R, T.right);
	  
	%else
	%  q.points_seen = q.points_seen +...
	%      size(T.right.indices, 2);
	%  q.samples_made = q.samples_made +...
	%      floor(beta * size(T.right.indices, 2));
	end
	
      else
	% go down the right subtree
	%display(sprintf('Going Right'));
	q = single_LMTree_time_constrained_search(q, R, T.right);
	
	% perform backtracking
	dSqToMargin = (class + T.left_min) *...
	    (class + T.left_min)...
	    / sum(T.omega .* T.omega);

	if dSqToMargin < q.ub
	  %display(sprintf('Right backtrack...'));
	  q = single_LMTree_time_constrained_search(q, R, T.left);
	  
	%else
	%  q.points_seen = q.points_seen +...
	%      size(T.left.indices, 2);
	%  q.samples_made = q.samples_made +...
	%      floor(beta * size(T.left.indices, 2));
	end
      end
    end
    %  end
    %end
%  end
%end
 
