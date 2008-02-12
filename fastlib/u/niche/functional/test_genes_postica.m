clear;
load gene_results;

% discriminant analysis using pc features %
% pc_scores is d x N

g1_indices = find(phases == g1_phase);
nong1_indices = find(phases ~=g1_phase & phases ~= unknown_phase);

used_scores = pc_scores;

used_scores = used_scores(1:5,:);

svm_data = [used_scores(:,g1_indices) used_scores(:,nong1_indices)]';
svm_labels = [1 * ones(length(g1_indices),1);
	      -1 * ones(length(nong1_indices),1)];

% write data so that we can efficiently call svmltrain many times
svmlwrite('funknet', svm_data, svm_labels);

% set initial svm options %
svm_options = ...
    svmlopt('Kernel', 2, 'KernelParam', .01, 'C', .2, 'ComputeLOO', 1, ...
	    'ExecPath','/home/niche/matlab/toolboxes/svml');
C = 1e-3;

C_epoch = 1;

while C < 100

  % set option for C (the regularization parameter)
  svm_options = ...
      svmlopt(svm_options, 'C', C);
  
  latestSVM = svml('latestSVM', svm_options);

  latestSVM = ...
      svmltrain(latestSVM, 'funknet');
  
  load loocv_error.txt
  
  C_array(C_epoch) = C;
  loocv_errors(C_epoch) = loocv_error;
  
  C = C * 1.01; % geometrically increase C
  C_epoch = C_epoch + 1;
end

% for i=1:size(svm_data, 1)
%   latestSVM = ...
%       svmltrain(latestSVM, [svm_data([1:(i-1) (i+1):end], :)], ...
% 		svm_labels([1:(i-1) (i+1):end]));
%   ypred(i) = svmlfwd(latestSVM, svm_data(i,:), svm_labels(i));
% end

% sum((2 * (ypred > 0) - 1) == svm_labels') / length(svm_labels) 