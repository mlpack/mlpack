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

target_data = responses(:, find(is_target));
nontarget_data = responses(:,find(~is_target));

selected_indices = shuffle(1:length(nontarget_data));
selected_indices = selected_indices(1:length(target_data));
nontarget_data = nontarget_data(:, selected_indices);

data = [target_data nontarget_data];



t = 1/240:1/240:1;

mybasis = create_bspline_basis([1/240 1], 30, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));





myfdPar = fdPar(mybasis, 2, 0);

myfd = data2fd(data, t, mybasis);

basis_curves = eval_basis(t, mybasis);
data_curves = basis_curves * getcoef(myfd);


myfdPar = fdPar(mybasis, 2, 1e-7);
pca_results = pca_fd(myfd, 30, myfdPar);

[ic_curves, ic_coef, Y, pc_coef, pc_curves, pc_scores, W] = ...
    funcica(t, myfd, 30, basis_curves, myfdPar, ...
	    basis_inner_products);

plot(t, -ic_curves(:,6));

%pc_curves = basis_curves * pc_coef;
%ic_curves = basis_curves * ic_coef_pos;

%}
