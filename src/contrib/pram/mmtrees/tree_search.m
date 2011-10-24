% The following function performs the recursion on the tree while
% searching for the nearest neighbor.
function [q] = tree_search(q, R, T, beta)

% either we are computing exact or the required number of leaves
% have not been visited
%if (q.compute_exact == 1 | q.ann_check > 0)

  if q.samples_made > q.samples_reqd
    q.points_seen = q.points_seen + size(T.indices,2);
    
  else
    % if leaf, do naive search
    if T.leaf == 1
      
      nPoints = size(T.indices, 2); 
      nSamples = 0;
      
      
      %[nDim nPoints] = size(leafData);
      
      %display(sprintf('Leaf: %d points', nPoints));
      % if leaf big, then just sample
      leafData = [];
      samplesIndices = [];
      if (q.compute_exact == 0) & (nPoints > 30)
	nSamples = min(ceil(nPoints * beta),...
		       min(q.samples_reqd - q.samples_made, 0));
	sampleIndices = randi([1 nPoints], nSamples, 1);
	leafData = R(:, T.indices(sampleIndices));
      else
	nSamples = nPoints;
	sampleIndices = [1:nPoints];
	leafData = R(:, T.indices);
      end      

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
	q.samples_made = q.samples_made + nSamples;
	q.points_seen = q.points_seen + nPoints;
	q.from_leaf = q.from_leaf + nSamples;
	
	q.num_leaves = q.num_leaves + 1;
      %end
      %if q.ann_check == 0
      %  q.ann_check = q.ann_check - 1;
      %end
    else
      
      % if this node is summarizable, just summarize it
      if (size(T.indices, 2) * beta <= 20) &...
	    (q.compute_exact == 0)
		
	q.num_summed = q.num_summed + 1;
	
	nPoints = size(T.indices, 2);
	
	% sampling required number of points from this node
	nSamples = ceil(nPoints * beta);
	sampleIndices = randi([1 nPoints], nSamples, 1);
	leafData = R(:, T.indices(sampleIndices));
	
	%[nDim nPoints] = size(leafData);
	
	%display(sprintf('Leaf: %d points', nPoints));

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
	  q.samples_made = q.samples_made + nSamples;
	  q.points_seen = q.points_seen + nPoints;
	  q.from_int_node = q.from_int_node + nSamples;
	  
	  %q.num_leaves = q.num_leaves + 1;
	%end
	
      else
	% the query is almost done, only needs a few more samples
	if (q.samples_reqd - q.samples_made <= 20) & ...
	      (q.compute_exact == 0)

	  q.almost_done = q.almost_done + 1;
	  
	  nPoints = size(T.indices, 2);
	  
	  % sampling required number of points from this node
	  nSamples = q.samples_reqd - q.samples_made; %ceil(nPoints * beta);
	  sampleIndices = randi([1 nPoints], nSamples, 1);
	  leafData = R(:, T.indices(sampleIndices));
	  
	  %[nDim nPoints] = size(leafData);
	  
	  %display(sprintf('Leaf: %d points', nPoints));

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
	    q.samples_made = q.samples_made + nSamples;
	    q.points_seen = q.points_seen + nPoints;
	    
	    %q.num_leaves = q.num_leaves + 1;
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
	    q = tree_search(q, R, T.left, beta);
	    
	    % perform backtracking
	    dSqToMargin = (T.right_min - class) *...
		(T.right_min - class)...
		/ sum(T.omega .* T.omega);

	    if dSqToMargin < q.ub
	      %display(sprintf('Left backtrack...'));
	      q = tree_search(q, R, T.right, beta);
	    else
	      q.points_seen = q.points_seen +...
		  size(T.right.indices, 2);
	      q.samples_made = q.samples_made +...
		  floor(beta * size(T.right.indices, 2));
	    end
	    
	  else
	    % go down the right subtree
	    %display(sprintf('Going Right'));
	    q = tree_search(q, R, T.right, beta);

	    % perform backtracking
	    dSqToMargin = (class + T.left_min) *...
		(class + T.left_min)...
		/ sum(T.omega .* T.omega);

	    if dSqToMargin < q.ub
	      %display(sprintf('Right backtrack...'));
	      q = tree_search(q, R, T.left, beta);
	    else
	      q.points_seen = q.points_seen +...
		  size(T.left.indices, 2);
	      q.samples_made = q.samples_made +...
		  floor(beta * size(T.left.indices, 2));
	    end
	  end
	end
      end
    end
  end
%end
 
