function [results_c] = drive_RPTree_expt(c_vals, taus, ...
					 alphas, Q, R, ...
					 rank_mat)
% Input variables:
% c_vals: The different values of c to build different RP trees
% tau_vals: The different values of rank-approximation to try on
% each tree
% alpha_vals: The different alpha values to be tried throughout the
% experiment
% Q: The query set
% R: The reference set
% rank_mat: The matrix containing the relative ranks of the
% reference points with respect to the queries (used to compute the
% error)
% flag: Whether the split-by-distance in the RPTrees are done on
% the original data set or the projected data set.
%
% Output:
% results_c: A cell containing the results for each tree
% corresponding to a value of c. For each c, the result will be an
% array with avg. DC, avg. MC, avg. rank error and max. rank error


flag = 0;

[nDim, nPoints] = size(R);
  for i = 1:length(c_vals)
    
    % form a set of random projections
    P = random_unit_vector(nDim, 50);
    
    % build the tree for this c value
    display(sprintf('Building tree for %0.2f....', c_vals(i))); 
    T = RPTree_build(R, P, c_vals(i), 1);
    
    nn_res = [];
    ind = 1;
    % for each alpha
    for j = 1:length(alphas)
      for k = 1:length(taus)
	[tmp1, tmp2, time] = RPTree_rann_search(Q, R, T, taus(k), ...
						alphas(j), ...
						rank_mat);
	nn_res(ind, :) = tmp2;
	ind = ind +1;
      end
    end
    
    results_c(i).a = nn_res;

    clear T;
    clear nn_res;
    
    if flag == 1
      display(sprintf('Part 2:'));
      
      T = RPTree_build(R, P, c_vals(i), 0);
      
      nn_res = [];
      ind = 1;
      % for each alpha
      for j = 1:length(alphas)
	for k = 1:length(taus)
	  [tmp1, tmp2, time] = RPTree_rann_search(Q, R, T, taus(k), ...
						  alphas(j), ...
						  rank_mat);
	  nn_res(ind, :) = tmp2;
	  ind = ind +1;
	end
      end
      
      results_c(i).b = nn_res;

      clear T;
      clear P;
      clear nn_res;
    end
    
  end
  