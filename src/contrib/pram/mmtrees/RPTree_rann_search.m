function [qRes, ann_accuracy, time] = RPTree_rann_search(Q, R, T, ...
						  tau_perc, alpha, rank_mat)

% performing search for every single query individually
[nDim, nQs] = size(Q);
ann_accuracy = [];
nRs = size(R, 2);
beta = 1;
compute_exact = 0;
tau = 0;

if tau_perc == 0
  compute_exact = 1;
else
  tau = floor(tau_perc * nRs / 100);
  beta = ComputeSampleSizes(tau, alpha, 1, nRs);
end

samples_reqd = ceil(beta* nRs);
display(sprintf('Tau: %d, Alpha: %0.2f, n:%d, N:%d', tau, alpha,...
		samples_reqd, nRs));

%error = nQs;
%numLeaf = 1;

%while error ~= 0
  
  %display(sprintf('Visiting %d leaves for ANN', numLeaf));
  time = 0;
  error_prob = 0;
  avg_rank_error = 0;
  max_error = -1.0;
  min_error = nRs + 1;

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
    q.compute_exact = compute_exact;
    
    q.points_seen = 0;
    q.samples_made = 0;
    q.samples_reqd = samples_reqd;
    %else
    % q.compute_exact = 0;
    %end
    
    q.num_summed = 0;
    q.from_leaf = 0;
    q.from_int_node = 0;
    q.almost_done = 0;
    
    tic;
    q = RPTree_search(q, R, T, beta);
    q.nn_time = toc;
    time = time + toc;

    if compute_exact == 1
      q.true_nn_dist = q.ub;
    end
    
    %if numLeaf == 1
    
    % computing true rank with the given rank matrix
    rank_error = rank_mat(i, q.nn) - 1;
    
    %if rank_error > 0
    %  display(sprintf('%d: %f', i, q.ub));
    %end
    
    
    avg_rank_error = avg_rank_error + rank_error;
    if rank_error > max_error
      max_error = rank_error;
    end
    if rank_error < min_error
      min_error = rank_error;
    end
    
    if (rank_error > tau)
      error_prob = error_prob + 1;
    end
  
    qRes(i) = q;
  end
  

  sample_stats = [qRes(:).points_seen; qRes(:).samples_made; ...
		  qRes(:).from_leaf; qRes(:).from_int_node; ...
		  qRes(:).num_leaves; qRes(:).num_summed; ...
		  qRes(:).almost_done];
  
  avg_sample_stats = mean(sample_stats, 2);
  
  
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
    ann_accuracy(1) = avg_stats(2);
    ann_accuracy(2) = avg_stats(3);
    ann_accuracy(3) = avg_rank_error / nQs;
    ann_accuracy(4) = max_error;
    ann_accuracy(5) = 1 - (error_prob / nQs);
    
    display(sprintf('Avg.time:%0.2f, Avg.DC:%0.2f, Avg.MC:%0.2f',...
        avg_stats(1), avg_stats(2), avg_stats(3)));
    display(sprintf('Avg.RE:%0.2f, Max.RE:%d, Min.RE:%d, alpha:%0.2f',...
        avg_rank_error/nQs, max_error, min_error,...
        1 - (error_prob/nQs)));
    
    %display(sprintf('Avg. points seen:%0.2f, Avg. Samples made:%0.2f',...
    %	    avg_sample_stats(1), avg_sample_stats(2)));
    %display(sprintf('Avg. from leaf:%0.2f, Avg. from int node:%0.2f',...
    %	    avg_sample_stats(3), avg_sample_stats(4)));
    %display(sprintf('Avg.leaves:%0.2f, Avg.summed:%0.2f, Avg.almost:%0.2f',...
    %	    avg_sample_stats(5), avg_sample_stats(6),...
    %       avg_sample_stats(7)));
    
    
    %display(mean([qRes(:).samples_made]));
    %display(sprintf('Avg. ANN time:%f, Avg. ANN DC:%f, Avg. ANN MC:%f',...
    %	    avg_stats(2), avg_stats(5), avg_stats(6)));
    %end
    %display(sprintf('Avg. ANN time:%f, Avg. ANN DC:%f, Avg. ANN MC:%f',...
    %	  avg_stats(2), avg_stats(5), avg_stats(6)));


    
    
% This is the function which computes the number of samples
% required to get the desired rank approximation
function [beta] = ComputeSampleSizes(tau, alpha, k, N)

lb = k;
ub = N;
n = ceil((lb + ub) / 2);
avoid_loop = 0;

%samples = ones(N,1);
done = 0;

while done == 0
  prob = ComputeProb(N, n, k, tau);
  %display(n);
  %display(prob);
  if (prob > alpha)
    if (prob - alpha < 0.001 | ub == lb + 1)
      done = 1;
    else
      ub = n;
    end
  else
    if (prob < alpha)
      if (n == lb)
	n = n+1;
	avoid_loop = 1;
      else
	lb = n;
      end
    else
      done = 1;
    end
  end
  if avoid_loop == 0
    n = ceil((ub + lb) / 2);
  else
    avoid_loop = 0;
  end
end

%ub = N;
%samples(ub) = n+1;
%display(n);
beta = (n+1) / N;


% This function computes the probability of the supposed order
% statistic given the number of samples.
function [prob] = ComputeProb(N, n, k, tau)

%for now just doing for k = 1
% computing P(d_(1) <= D_(1+tau))
temp_b = zeros(tau+1, 1);
temp_b(tau+1) = 1;

j = 1;
for i=1:tau
  frac = (N - (n-1) -j) / (N - j);
  temp_b(tau+1 - i) = temp_b(tau+2 -i) * frac;
  j = j+1;
end

prob = (n / N) * sum(temp_b, 1);
     

%end of file 