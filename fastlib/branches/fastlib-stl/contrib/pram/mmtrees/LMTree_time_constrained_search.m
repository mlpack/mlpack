function [ann_accuracy] = LMTree_time_constrained_search(Q, R, T, ...
						  rank_mat)

% performing search for every single query individually
[nDim, nQs] = size(Q);
ann_accuracy = [];
nRs = size(R, 2);

time = 0;
max_num_of_leaves = 0;

rank_mat_too_big = 0;
rank_fp = [];

if isa(rank_mat, 'char')
  rank_mat_too_big = 1;
  rank_fp = fopen(rank_mat, 'r');
  display(sprintf('Rank matrix too big.\n'));
end

qRes = [];
perc_done = 1;
done_sky = 1;

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
  
  q.points_seen = 0;
  q.rank_vec = [];
  %tic;
  if rank_mat_too_big == 0
    q.rank_vec = rank_mat(i, :);
  else
    rank_str = fgetl(rank_fp);
    %rank_ind = 0;
    %while true
    %  [rank, rank_str] = strtok(rank_str, ',');
    %  if isempty(rank), break; end
    %  rank_ind = rank_ind + 1;
    %  q.rank_vec(rank_ind) = str2num(rank);
    %end
    q.rank_vec = [str2num(rank_str)];
    
    if size(q.rank_vec, 2) ~= nRs
      display(sprintf('WTF: %d <-> %f\n', nRs, rank_ind));
    end
  end
  q.rank_error_list = [];
  q.dist_comp_list = [];
  %read_time = toc;
  
  %display(sprintf('#%d: %0.2f', i, read_time));
  
  tic;
  q = single_LMTree_time_constrained_search(q, R, T);
  q.nn_time = toc;
  time = time + toc;
  
  if q.num_leaves > max_num_of_leaves
    max_num_of_leaves = q.num_leaves;
  end
  
  qRes(i).nn_dist_comp = q.nn_dist_comp;
  qRes(i).nn_marg_comp = q.nn_marg_comp;
  qRes(i).rank_error_list = q.rank_error_list;
  qRes(i).dist_comp_list = q.dist_comp_list;
    
  clear q;
  
  pdone = i * 100 / nQs;
  
  if pdone >= done_sky * perc_done
    if (done_sky > 1) 
      fprintf(1, '\b\b\b=%d%%', int16(pdone));
    else
      fprintf(1, '=%d%%', int16(pdone));
    end
    done_sky = done_sky + 1;
  end
   
end
pdone = i * 100 / nQs;

if pdone >= done_sky * perc_done
  if (done_sky > 1) 
    fprintf(1, '=%d%%', int16(pdone));
  else
    fprintf(1, '=%d%%', int16(pdone));
  end
  %done_sky = done_sky + 1;
end
fprintf(1, '\n');


% displaying stats for exact search
comp_stats = [qRes(:).nn_dist_comp; ...
	      qRes(:).nn_marg_comp]';
avg_stats = mean(comp_stats, 1);
display(sprintf('NN DC: %0.2f, NN MC: %0.2f',...
		avg_stats(1), avg_stats(2)));

% computing the rank errors for the number of leaves seen and also
% computing the squared rank errors to compute mean and standard
% deviation

display(sprintf('Max. number of leaves: %d', ...
		max_num_of_leaves));

rank_error_list = zeros(max_num_of_leaves, 1);
sq_rank_error_list = zeros(max_num_of_leaves, 1);
max_rank_error_list = -1 * ones(max_num_of_leaves, 1);
min_rank_error_list = Inf * ones(max_num_of_leaves, 1);

for i = 1:nQs
  for j = 1:length(qRes(i).rank_error_list)
    rank_error_list(j) = rank_error_list(j) + ...
	qRes(i).rank_error_list(j);

    sq_rank_error_list(j) = sq_rank_error_list(j) + ...
	(qRes(i).rank_error_list(j))^2;
    
    if qRes(i).rank_error_list(j) > max_rank_error_list(j)
      max_rank_error_list(j) = qRes(i).rank_error_list(j);
    end
    
    if qRes(i).rank_error_list(j) < min_rank_error_list(j)
      min_rank_error_list(j) = qRes(i).rank_error_list(j);
    end
  end
end

mean_rank_error_list = rank_error_list / nQs;

std_rank_error_list = (nQs*sq_rank_error_list - rank_error_list.^2).^0.5 ...
    / nQs;

ann_accuracy.means = mean_rank_error_list;
ann_accuracy.stds = std_rank_error_list;
ann_accuracy.min = min_rank_error_list;
ann_accuracy.max = max_rank_error_list;

if rank_mat_too_big == 1
  fclose(rank_fp);
end

%end of file 