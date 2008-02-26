

rest_of_data = responses(:,5221:end);

rest_of_data_myfd = data2fd(rest_of_data, t, mybasis);

mean_result = pca_fd(rest_of_data_myfd, 0);

centered_rest_of_data_coef = ...
    getcoef(rest_of_data_myfd) - ...
    repmat(getcoef(mean_result.meanfd), ...
	   1, ...
	   size(getcoef(rest_of_data_myfd), 2));

centered_rest_of_data_myfd = fd(centered_rest_of_data_coef, mybasis);

rest_ic_scores = ...
    get_scores(centered_rest_of_data_coef, ...
	       ic_coef, ...
	       basis_inner_products)';


% let the 'target component' be defined as the component identified
% to most resemble the P300 response
    
 % for each set of 12 trials, pick the trial whose target component
 % score is the largest
 
% check classification accuracy by using is_target

used_ic_scores = rest_ic_scores;
used_is_target = is_target(5221:end);


for i = 1:size(ic_curves,2)
  scale_up_factor = ...
      1 / sqrt(sum(sum((ic_coef(:,i) * ic_coef(:,i)') .* ...
		       basis_inner_products)));
  ic_coef(:,i) = scale_up_factor * ic_coef(:,i);
  ic_curves(:,i) = scale_up_factor * ic_curves(:,i);
  used_ic_scores(i,:) = scale_up_factor * used_ic_scores(i,:);
end

mean_target_scores = ...
    mean(used_ic_scores(:, find(used_is_target))');
mean_nontarget_scores = ...
    mean(used_ic_scores(:, find(~used_is_target))');

mean_diff_scores = mean_target_scores - mean_nontarget_scores;


[y,ic] = max(abs(mean_diff_scores));
ic = 6;
the_sign = sign(mean_diff_scores(ic));

first = [1 7];
last = [6 12];

epochs_to_test = 56;

correct = zeros(2, epochs_to_test);

a = [];
a_val = [];
a_i = [];
a_rank = [];

num_trials_used = 15;%num_trials;

for k = 1:2
  for j = 0:(epochs_to_test-1)
    for i = 0:(num_trials_used-1);
      a(i+1,:) = ...
	  the_sign * ...
	  used_ic_scores(ic,((15*12*j) + (12*i) + first(k)):((15*12*j) + (12*i) + last(k)));
    end
    
    [a_val,a_i] = sort(a, 2, 'ascend'); % ascending -> higher = better
    
    for i = 1:num_trials_used
      a_rank(i,a_i(i,:)) = 1:6;
    end
    
    %fprintf('j = %d\n', j);
    
    result = ...
	[used_is_target(((15*12*j) + (12*0) + first(k)):((15*12*j) + (12*0) + last(k))); ...
	 mean(a_rank); ...
	 std(a)];
    
    true_arg_max = find(result(1,:) == 1);
    [max_val, arg_max] = max(result(2,:));
    
    if true_arg_max == arg_max
      correct(k,j+1) = 1;
    end
  end
end
  
for k = 1:2
  if k == 1
    fprintf('\n\nColumn Selection Performance\n');
  else
    fprintf('\n\nRow Selection Performance\n');
  end
  
  fprintf('%d correct out of %d\n', sum(correct(k,:)), epochs_to_test);
  fprintf('accuracy: %f\n', sum(correct(k,:)) / epochs_to_test);
  
  fprintf('incorrect indices\n');
  fprintf('%d,', find(~correct(k,:)));
  fprintf('\n');
end

fprintf('\n\nOverall Performance\n');
fprintf('%d correct letter selections\n', sum(correct(1,:) & ...
					      correct(2,:)));
fprintf('accuracy: %f\n', sum(correct(1,:) & correct(2,:)) / epochs_to_test);
