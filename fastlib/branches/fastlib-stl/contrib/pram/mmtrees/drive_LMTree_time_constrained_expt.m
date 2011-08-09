function [ann_c] = drive_LMTree_time_constrained_expt(C_vals, ...
						  Q, R, rank_mat, ...
						  output_file)

% Input variables:
% c_vals: The different values of c to build different RP trees
% Q: The query set
% R: The reference set
% rank_mat: The matrix containing the relative ranks of the
% reference points with respect to the queries (used to compute the
% error)
% flag: Whether the split-by-distance in the RPTrees are done on
% the original data set or the projected data set.
%
% Output:
% ann_c: A cell containing the results for each tree
% corresponding to a value of c. For each c, the result will be an
% array with avg. DC, avg. MC, avg. rank error and max. rank error

  [nDim, nPoints] = size(R);
  for i = 1:length(C_vals)
    
    % build the tree for this c value
    display(sprintf('Building tree for %f....', C_vals(i))); 
    T = CPMMC_tree(R, C_vals(i), 1e-6, 0);
    
    % run the time constrained algorithm
    display(sprintf('Running the search algorithm...'))
    ann_c(i).a = LMTree_time_constrained_search(Q, R, T, rank_mat);
    ann_c(i).c = C_vals(i);
    clear T;

  end

  save(output_file, 'ann_c');
  