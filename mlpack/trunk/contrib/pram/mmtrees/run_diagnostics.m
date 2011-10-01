function run_diagnostics(given_vals, est_vals)

nPoints = size(given_vals,1);
total_comp = sum(given_vals(:,2));
min_comp = sum(given_vals(:,1));

max_possible_gain = total_comp - min_comp;

error_ind = find(given_vals(:,1) > est_vals(:,1));
error_ind_p = find(given_vals(:,1) > est_vals(:,2));

num_error = size(error_ind, 1);
num_error_p = size(error_ind_p, 1);

total_gain_ind = find(given_vals(:,2) > est_vals(:,1));
correct_gain_ind = setdiff(total_gain_ind, error_ind);

num_correct_gain = size(correct_gain_ind, 1);

display(sprintf('# Points: %d, #Error: %d, %f, #Correct Gains:%d, %f', ...
		nPoints, num_error, num_error / nPoints, ...
		num_correct_gain, num_correct_gain / nPoints));

display(sprintf('#Error with mean+2*std: %d, %f', num_error_p, ...
		num_error_p / nPoints));

display(sprintf('Min Comp: %f, Max possible gain: %f', min_comp / ...
		total_comp, max_possible_gain / total_comp));

correct_gain = sum(given_vals(correct_gain_ind, 2) - ...
		   est_vals(correct_gain_ind, 1));

display(sprintf('Correct gain: %f', correct_gain / total_comp));

wrong_loss = sum(given_vals(error_ind, 1) - est_vals(error_ind, ...
						  1));
display(sprintf('Wrong loss: %f', wrong_loss / ...
		sum(given_vals(error_ind,1))));
