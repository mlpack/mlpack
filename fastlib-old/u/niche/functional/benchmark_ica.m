% load data

% cz = 11;

data_dir = ['../../../../neurofunk/' ...
	    'bci_comp_III_Wads_2004/' ...
	    'BCI_Comp_III_Wads_2004/'];


TargetChar_filename = [data_dir 'TargetChar.mat'];
responses_filename = [data_dir 'filtered_responses.mat'];


% load and do inits for TargetChar data

TargetChar = load(TargetChar_filename);
TargetChar = TargetChar.TargetChar;

% map each letter to a [row column] pair

screen = char('A','B','C','D','E','F', ...
	      'G','H','I','J','K','L', ...
	      'M','N','O','P','Q','R', ...
	      'S','T','U','V','W','X', ...
	      'Y','Z','1','2','3','4', ...
	      '5','6','7','8','9','_');

screen = reshape(screen, 6, 6)';





		    
% load and do inits for responses data

responses = load(responses_filename);
responses = responses.filtered_responses;

[num_epochs, num_flashes, num_trials, T] = size(responses);


% tag with target/nontarget
is_target = false(num_epochs, num_flashes, num_trials);
for epoch = 1:num_epochs
  target_letter = TargetChar(epoch);
  [r, c] = find(screen == target_letter);
  target_pair = [r c] + [6 0];

  for stimulus_code = 1:num_flashes
    for trial = 1:num_trials
      if ~isempty(find(target_pair == stimulus_code))
	is_target(epoch, stimulus_code, trial) = true;
      end
    end
  end
end

new_responses = zeros(T, num_flashes, num_trials, num_epochs);
new_is_target = zeros(num_flashes, num_trials, num_epochs);
for i = 1:T
  i
  for j = 1:num_flashes
    j
    for k = 1:num_trials
      k
      for l = 1:num_epochs
	l
	new_responses(i,j,k,l) = responses(l,j,k,i);
	new_is_target(j,k,l) = is_target(l,j,k);
      end
    end
  end
end

responses = new_responses;
is_target = new_is_target;
clear new_responses;
clear new_is_target;

responses = ...
    reshape(responses, [T, num_flashes * num_trials * num_epochs]);

is_target = ...
    reshape(is_target, [1 num_flashes * num_trials * num_epochs]);


%{
target_data = responses(:, find(is_target));
nontarget_data = responses(:,find(~is_target));

selected_indices = shuffle(1:length(nontarget_data));
selected_indices = selected_indices(1:length(target_data));
nontarget_data = nontarget_data(:, selected_indices);

data = [target_data nontarget_data];
%}

data = responses(:,1:(num_flashes*num_trials*29));

[coeff, score, latent] = princomp(data');

p_small = min(find(cumsum(latent) / sum(latent) >= 0.9));

p_small = min(p_small, 12);

[Yopt, Wopt] = RADICAL(score(:,1:p_small)');



%{
t = 1/240:1/240:1;

m = 120;
p = 11;

mybasis = create_bspline_basis([1/240 1], m, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));



myfd = data2fd(data, t, mybasis);
mean_result = pca_fd(myfd, 0);
centered_data_coef = ...
    getcoef(myfd) - ...
    repmat(getcoef(mean_result.meanfd), 1, size(getcoef(myfd), 2));
centered_myfd = fd(centered_data_coef, mybasis);

centered_data_curves = basis_curves * getcoef(centered_myfd);

lambda = 3e-6;
myfdPar = fdPar(mybasis, 2, lambda);
pca_results = pca_fd(centered_myfd, p, myfdPar);
%return;
[ic_curves, ic_coef, ic_scores, ...
 pc_coef, pc_curves, pc_scores, W] = ...
    funcica(t, centered_myfd, p, basis_curves, myfdPar, ...
	    basis_inner_products);
%}

%plot(t, -ic_curves(:,6));

%for i = 1:size(ic_curves,2)
%  scale_up_factor = ...
%      1 / sqrt(sum(sum((ic_coef(:,i) * ic_coef(:,i)') .* ...
%		       basis_inner_products)));
%  ic_coef(:,i) = scale_up_factor * ic_coef(:,i);
%  ic_curves(:,i) = scale_up_factor * ic_curves(:,i);
%  ic_scores(i,:) = scale_up_factor * ic_scores(i,:);
%end

%save p300_filtered_lambda3Eneg6_correct_results_120.mat;


% given a set of curves, identify component curves of variation
% once we have these curves, see how well each curve differentiates
% the data

% decision rule - given 6 curves (corresponding to rows or
% columns), pick the curve with the greatest score for a particular
% independent component curve
% consider each trial as indepedent and take a vote over the 1st k
% trials for k <= 15

% re-run experiments with gene expression data using correct ic
% scores




% compute principal components

% do dimensionality reduction

% use RADICAL to compute independent components

% transform components to unit components

% re-use ICAFunc classifier that selects item with maximum score
% over 15 trials

