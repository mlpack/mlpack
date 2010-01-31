

rest_of_data = responses(:,5221:end);

centered_rest_of_data = ...
    rest_of_data - ...
    repmat(mean(rest_of_data, 2), 1, size(rest_of_data, 2));

rest_score = centered_rest_of_data' * coeff;

rest_ic_scores = WoptNormed * rest_score(:, 1:p_small)';


used_ic_scores = rest_ic_scores;
used_is_target = is_target(5221:end);


%emp_waveform = mean(responses(:, find(is_target)), 2);

%inc = 35;
%smooth_emp_waveform = ...
%    ppval(spline(t(1:inc:end), emp_waveform(1:inc:end)), t);

%used_ic_scores(6,:) = ...
%    smooth_emp_waveform * centered_rest_of_data;


% conclusions for smoothing the empirical waveform: no amount of
% smoothing provides smoothness in the 'right places'

% conclusions for ICA
% in the best case
% IC    col      row      letter
% 6     30.4%    51.8%    16.1%
% 10    35.7%    48.2%    17.9%

mean_target_scores = ...
    mean(used_ic_scores(:, find(used_is_target))');
mean_nontarget_scores = ...
    mean(used_ic_scores(:, find(~used_is_target))');

mean_diff_scores = mean_target_scores - mean_nontarget_scores;


[y,ic] = max(abs(mean_diff_scores));
ic = 3;
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
	 mean(a); ...
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
