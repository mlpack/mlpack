function [qRes, ann_accuracy, time] = CPMMC_tree_search(Q, R, T, FName)

% performing search for every single query individually
[nDim, nQs] = size(Q);
ann_accuracy = [];

%error = nQs;
%numLeaf = 1;

%while error ~= 0
  
  %display(sprintf('Visiting %d leaves for ANN', numLeaf));
  time = 0;
  error = 0;
  avg_eps = 0.0;
  max_eps = -1.0;

  for i = 1:nQs
    q.p = Q(:, i);
    q.ub = Inf;
    q.nn = -1;
    % q.ann = -1;
    q.num_leaves = 0;
    %q.ann_dist = Inf;
    q.nn_time = 0;
    %q.ann_time = 0;
    q.nn_dist_comp = 0;
    q.nn_marg_comp = 0;
    %q.ann_dist_comp = 0;
    %q.ann_marg_comp = 0;
    
    %if numLeaf == 1
    q.compute_exact = 1;
    %else
    % q.compute_exact = 0;
    %end
    
    
    tic;
    [q, temp] = tree_search(q, R, T, ann_accuracy);
    q.nn_time = toc;
    time = time + toc;

    q.true_nn_dist = q.ub;

    %if numLeaf == 1
    qRes(i) = q;
    %  ub = q.ub;
    %else
    %  ub = qRes(i).ub;
    %  qResTemp(i) = q;
    %end
        
    %if ub < q.ann_dist
    %  error = error + 1;
    %  epsilon = (q.ann_dist / ub) -1;
    %  avg_eps = avg_eps + epsilon;
    %  if epsilon > max_eps
    %max_eps = epsilon;
    %  end
    %end
    
    % clear q;
  end
  

  % avg_eps = avg_eps / nQs;

  %display(sprintf('%d: E:%d/%d, AEps:%f, MEps:%f', ...
  %	    numLeaf, error, nQs, avg_eps, max_eps));

  % if numLeaf == 1
  comp_stats = [qRes(:).nn_time;...
  %qRes(:).ann_time;...
		qRes(:).nn_dist_comp;...
		qRes(:).nn_marg_comp;...
  %qRes(:).ann_dist_comp;...
  %qRes(:).ann_marg_comp...
	       ]';

  avg_stats = mean(comp_stats, 1);
  %else 
  %  comp_stats = [qResTemp(:).nn_time;...
	%	  qResTemp(:).ann_time;...
	%	  qResTemp(:).nn_dist_comp;...
	%	  qResTemp(:).nn_marg_comp;...
	%	  qResTemp(:).ann_dist_comp;...
	%	  qResTemp(:).ann_marg_comp]';

    %avg_stats = mean(comp_stats, 1);
    %end
  
    %if numLeaf == 1
    % display(sprintf('%d: E:%d/%d, AEps:%f, MEps:%f', ...
    %	    numLeaf, error, nQs, avg_eps, max_eps));
    display(sprintf('Avg. NN time:%f, Avg. NN DC:%f, Avg. NN MC:%f',...
		    avg_stats(1), avg_stats(2), avg_stats(3)));
    %display(sprintf('Avg. ANN time:%f, Avg. ANN DC:%f, Avg. ANN MC:%f',...
    %	    avg_stats(2), avg_stats(5), avg_stats(6)));
    %end
    %display(sprintf('Avg. ANN time:%f, Avg. ANN DC:%f, Avg. ANN MC:%f',...
    %	  avg_stats(2), avg_stats(5), avg_stats(6)));

    last_dc_val_sum = 0; 
    last_mc_val_sum = 0;
    
    for i = 1:nQs
      %qRes(i).p = Q(:, i);
      qRes(i).ub = Inf;
      qRes(i).nn = -1;
      % q.ann = -1;
      qRes(i).num_leaves = 1;
      %q.ann_dist = Inf;
      qRes(i).nn_time = 0;
      %q.ann_time = 0;
      qRes(i).nn_dist_comp = 0;
      qRes(i).nn_marg_comp = 0;
      %q.ann_dist_comp = 0;
      %q.ann_marg_comp = 0;
    
      %if numLeaf == 1
      qRes(i).compute_exact = 0;
      %else
      % q.compute_exact = 0;
      %end
    
      % setting up the last vals
      qRes(i).dc_sum = last_dc_val_sum;
      qRes(i).mc_sum = last_mc_val_sum;
    
      tic;
      [qRes(i), ann_accuracy] = tree_search(qRes(i), R, T, ann_accuracy);
      qRes(i).nn_time = toc;
      time = time + toc;

      while qRes(i).num_leaves <= size(ann_accuracy, 1)
	ann_accuracy(qRes(i).num_leaves, 2) = ...
	    ann_accuracy(qRes(i).num_leaves, 2) + ...
	    qRes(i).nn_dist_comp;
	
	ann_accuracy(qRes(i).num_leaves, 3) = ...
	    ann_accuracy(qRes(i).num_leaves, 3) + ...
	    qRes(i).nn_marg_comp;
	
	qRes(i).num_leaves = qRes(i).num_leaves + 1;
      end
      
	
      
      last_dc_val_sum = last_dc_val_sum + qRes(i).nn_dist_comp;
      last_mc_val_sum = last_mc_val_sum + qRes(i).nn_marg_comp;

      %q.true_nn_dist = q.ub;

      %if numLeaf == 1
      %qRes(i) = q;
      %  ub = q.ub;
      %else
      %  ub = qRes(i).ub;
      %  qResTemp(i) = q;
      %end
      
      %if ub < q.ann_dist
      %  error = error + 1;
      %  epsilon = (q.ann_dist / ub) -1;
      %  avg_eps = avg_eps + epsilon;
      %  if epsilon > max_eps
      %max_eps = epsilon;
      %  end
      %end
    
      % clear q;
    end
  

  %avg_eps = avg_eps / nQs;

  ann_accuracy = ann_accuracy / nQs;
  %display(sprintf('%d,%g,%g', error, avg_stats(5), avg_stats(6)));
  %numLeaf = numLeaf + 1;
  %clear qResTemp;
%end

csvwrite(FName, ann_accuracy);



% The following function performs the recursion on the tree while
% searching for the nearest neighbor.
function [q, ann_data] = tree_search(q, R, T, ann_data)

% either we are computing exact or the required number of leaves
% have not been visited
%if (q.compute_exact == 1 | q.ann_check > 0)

  % if leaf, do naive search
  if T.leaf == 1
    leafData = R(:, T.indices);
    
    [nDim nPoints] = size(leafData);
    
    %display(sprintf('Leaf: %d points', nPoints));

    diffMat = leafData - repmat(q.p, [1 nPoints]);
    diffSqMat = diffMat .* diffMat;
    
    distSqVec = sum(diffSqMat, 1);
    
    [dist ind] = min(distSqVec);
    
    q.nn_dist_comp = q.nn_dist_comp + nPoints;
    
    if dist < q.ub
      q.ub = dist;
      q.nn = T.indices(ind);
      
      %if q.ann_check > 0
	%q.ann = T.indices(ind);
	%q.ann_dist = dist;
      %end
    end
    
    if q.compute_exact == 0
      % here we do the check for ann stats
      % q.ann_check = q.ann_check - 1;
      %q.ann_dist_comp = q.nn_dist_comp;
      %q.ann_marg_comp = q.nn_marg_comp;
      %q.ann_time = toc;
      
      % check if size(ann_data, 1) >= q.num_leaves
      
      if (size(ann_data, 1) < q.num_leaves)
	% here we have to insert a new element in the array
	
	% DEBUG_ASSERT(size(ann_data, 1) == q.num_leaves -1)
	if q.true_nn_dist < q.ub
	  ann_data(q.num_leaves, :) = [1, ...
		    q.dc_sum + q.nn_dist_comp, ...
		    q.mc_sum + q.nn_marg_comp];
	else
	  ann_data(q.num_leaves, :) = [0, ...
		    q.dc_sum + q.nn_dist_comp, ...
		    q.mc_sum + q.nn_marg_comp];
	end	
      else
	% just add your values to the existing element in the array
	if q.true_nn_dist < q.ub
	  ann_data(q.num_leaves, 1) = ...
	      ann_data(q.num_leaves, 1) + 1;
	end
	ann_data(q.num_leaves, 2) = ...
	    ann_data(q.num_leaves, 2) + q.nn_dist_comp;
	ann_data(q.num_leaves, 3) = ...
	    ann_data(q.num_leaves, 3) + q.nn_marg_comp;
      end
      
      q.num_leaves = q.num_leaves + 1;
    end
    %if q.ann_check == 0
    %  q.ann_check = q.ann_check - 1;
    %end
  else
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
      [q, ann_data] = tree_search(q, R, T.left, ann_data);
      
      % perform backtracking
      dSqToMargin = (T.right_min - class) * (T.right_min - class)...
	  / sum(T.omega .* T.omega);

      if dSqToMargin < q.ub
	%display(sprintf('Left backtrack...'));
	[q, ann_data] = tree_search(q, R, T.right, ann_data);
      end
      
    else
      % go down the right subtree
      %display(sprintf('Going Right'));
      [q, ann_data] = tree_search(q, R, T.right, ann_data);

      % perform backtracking
      dSqToMargin = (class + T.left_min) * (class + T.left_min)...
	  / sum(T.omega .* T.omega);

      if dSqToMargin < q.ub
	%display(sprintf('Right backtrack...'));
	[q, ann_data] = tree_search(q, R, T.left, ann_data);
      end
    end
  end
%end
 
