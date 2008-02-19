% cz = 11;

data_dir = ['../../../../neurofunk/' ...
	    'bci_comp_III_Wads_2004/' ...
	    'BCI_Comp_III_Wads_2004/'];


TargetChar_filename = [data_dir 'TargetChar.mat'];
responses_filename = [data_dir 'responses.mat'];


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
responses = responses.responses;

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


responses = ...
    reshape(responses, [num_epochs * num_flashes * num_trials, ...
		    T])';


is_target = ...
    reshape(is_target, [num_epochs * num_flashes * num_trials, ...
		    1])';
%{
target_data = responses(:, find(is_target));
nontarget_data = responses(:,find(~is_target));

selected_indices = shuffle(1:length(nontarget_data));
selected_indices = selected_indices(1:length(target_data));
nontarget_data = nontarget_data(:, selected_indices);

data = [target_data nontarget_data];
%}

data = responses(:,1:5100);


t = 1/240:1/240:1;

mybasis = create_bspline_basis([1/240 1], 30, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));





myfdPar = fdPar(mybasis, 2, 0);

myfd = data2fd(data, t, mybasis);
mean_result = pca_fd(myfd, 0);
centered_data_coef = ...
    getcoef(myfd) - ...
    repmat(getcoef(mean_result.meanfd), 1, size(getcoef(myfd), 2));
centered_myfd = fd(centered_data_coef, mybasis);

centered_data_curves = basis_curves * getcoef(centered_myfd);


myfdPar = fdPar(mybasis, 2, 1e-7);
pca_results = pca_fd(centered_myfd, 30, myfdPar);

[ic_curves, ic_coef, ic_scores, ...
 pc_coef, pc_curves, pc_scores, W] = ...
    funcica(t, centered_myfd, 30, basis_curves, myfdPar, ...
	    basis_inner_products);

%plot(t, -ic_curves(:,6));

for i = 1:size(ic_curves,2)
  scale_up_factor = ...
      1 / sqrt(l2_fnorm(t, ic_curves(:,i), ic_curves(:,i)));
  ic_curves(:,i) = scale_up_factor * ic_curves(:,i);
  ic_scores(i,:) = scale_up_factor * ic_scores(i,:);
end

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


