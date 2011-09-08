function [T, done, treeInfo] = ...
    CPMMC_tree(data, C, initVal, done, treeInfo, indices)
% Using CPMMC to hiearchically splitting the data
% forming it into a tree.
%
% 'data' should be column major, with every column 
% denoting a point.

%T.data = data; 
if nargin < 4
  indices = 1:size(data,2);
  fprintf(1, 'Building tree with %d points:   ', ...
	  length(indices));
  done = 0;
  treeInfo.numKdSplits = 0;
  treeInfo.totalSplits = 0;
  treeInfo.numLeaves = 0;
  treeInfo.leafSizes = 0;
  treeInfo.kdSplitWts = 0;
  treeInfo.totalWts = 0;
end

%l_init = 0.001 * size(data,2);
l_init = 0.001 * length(indices);

T.indices = indices;
T.leaf = 0;
T.left = [];
T.right = [];
T.omega = [];
T.b = [];
T.time = [];
temp_initVal = initVal;

% if the data size is small, treat it as a leaf
% if size(data, 2) < 30
if length(indices) <= 30
  T.leaf = 1;
  done = done + length(indices);
  treeInfo.numLeaves = treeInfo.numLeaves + 1;
  treeInfo.leafSizes = treeInfo.leafSizes + length(indices);
  
% Step 1: Find the best split
% Step 2: Partition the data using the split
% Step 3: Recurse on each partition
else
  nodeData = data(:, indices);
  
  l = l_init; % balance constraint
  b_param = 0; % flag to decide if we reduce value of l
  numTries = 0;
  C0 = C;
  left_ind = [];
  right_ind = [];
  time = 0;
  
  while (b_param == 0 & numTries < 10)
    numTries = numTries + 1;
    [omega, b, time] = CPMMC_split(nodeData, C0, l, initVal);
    %[omega, b, time] = CP_CCCP_split(nodeData, C0, l, initVal);
    
    f_theta = omega' * nodeData + b;
    
    left_ind = find( f_theta < 0 );
    right_ind = find( f_theta >= 0);
    
    outOfMargin_ind = find( abs(f_theta) >= 1); 
    
    numOutOfMargin = size(outOfMargin_ind, 2);
    
    minChildSize = min(size(left_ind, 2), size(right_ind, 2));
    if numOutOfMargin < 5
      if minChildSize / length(indices) < 0.25
	%keyboard;
	b_param = 0;
	l = l / 1.5;
      else
	b_param = 1;
      end
      %display(sprintf('%d, %d, %d, %f, %d', minChildSize, length(indices), ...
      %	      b_param, l, numTries));
    else
      b_param = 1;
      %display(sprintf('%d, %d, %d, %f, %d', minChildSize, size(data, 2), ...
      %      b_param, l, numTries));
    end
    
    if mod(numTries,10) == 0
      l = l_init;
      C0 = C0 / 1.5;
    end
    
    if mod(numTries, 50) == 0
      C0 = C0 / 10;
      l = l_init;
      %initVal = initVal / 5;
    end
    
  end  

  % test_flag = 0;  
  % if no split found, make it a leaf
  % POTENTIAL TWEAKS
  % Or we can just do bunch of kdtree splits to bring the number of
  % points in the leaves down to 30 or less
  if (length(left_ind) == 0) | (length(right_ind) == 0)
    %T.omega = [];
    %T.b = [];
    % T.leaf = 1;
    %keyboard;
    % This is where you start doing the kd-tree split
    % Think about how you are going to do it
    % test_flag = 1;
    % if test_flag == 0
    [omega, b, tmp_time] = KDTree_median_split(nodeData);
    f_theta = omega' * nodeData + b;
    left_ind = find( f_theta < 0 );
    right_ind = find( f_theta >= 0 );
    time = time + tmp_time;
    % end
    %display(sprintf('KDTree splits...'));
    %keyboard;
    
    treeInfo.kdSplitWts = treeInfo.kdSplitWts + length(indices);
    treeInfo.numKdSplits = treeInfo.numKdSplits + 1;
  end


  %left_data = data(:, left_ind);
  %right_data = data(:, right_ind);
  left_indices = indices(left_ind);
  right_indices = indices(right_ind);
  
  T.omega = omega;
  T.b = b;
  T.time = time;
  
  treeInfo.totalSplits = treeInfo.totalSplits + 1; 
  treeInfo.totalWts = treeInfo.totalWts + length(indices);
  
  %display(sprintf('%d : %d / %d, %0.3f, %d, Numtries:%d',...
	%	  length(indices), length(left_indices),...
	%	  length(right_indices), omega' * omega, ...
	%	  numOutOfMargin, numTries));
  
  %keyboard;
  
%if test_flag == 0
  %T.left_min = 0.0;
  %T.right_min = 0.0;
  
  %else
    
  % TWEAK: 
  % Tighter margins 
  T.left_min = min(abs(T.omega'*data(:, left_indices) + T.b));
  T.right_min = min(abs(T.omega'*data(:, right_indices) + T.b));
    
  %display(sprintf('Min: Left:%f, Right:%f',...
  %	    T.left_min, T.right_min));
  initVal = temp_initVal;
    
  [T.left, done, treeInfo]...
      = CPMMC_tree(data, C, initVal, done, ...
		   treeInfo, left_indices);
  
  pdone = floor(done * 100 / size(data,2));
  if pdone < 10
    fprintf(1, '\b\b%d%%', pdone); 
  else
    fprintf(1, '\b\b\b%d%%', pdone);
  end
  
  [T.right, done, treeInfo]...
      = CPMMC_tree(data, C, initVal, done, ...
		   treeInfo, right_indices);
  
%end
    
end

if nargin < 4
  fprintf(1, '\n');
  fprintf(1, 'Number of Leaves: %d, Avg. Leaf Size: %f\n',...
	  treeInfo.numLeaves, ...
	  treeInfo.leafSizes / treeInfo.numLeaves);
  fprintf(1, 'Total number of splits: %d, %% kd-splits: %f\n', ...
	  treeInfo.totalSplits, treeInfo.numKdSplits * 100 / ...
	  treeInfo.totalSplits);
  fprintf(1, 'Weighted %% kd-tree Splits: %f\n', ...
	  treeInfo.kdSplitWts * 100 / treeInfo.totalWts);
end
