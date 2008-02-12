% discriminant analysis using pc features %
% pc_scores is d x N

g1_indices = find(phases == g1_phase);
nong1_indices = find(phases ~=g1_phase & phases ~= unknown_phase);

ic_scores = ic_scores(1:5,:);

svm_data = [ic_scores(:,g1_indices) ic_scores(:,nong1_indices)]';
svm_labels = [1 * ones(length(g1_indices),1);
	      -1 * ones(length(nong1_indices),1)];


latestSVM = svml('latestSVM','Kernel',2,'KernelParam',.01,'C',.2, ...
		 'ExecPath','/home/niche/matlab/toolboxes/svml');

for i=1:size(svm_data, 1)
  latestSVM = ...
      svmltrain(latestSVM, [svm_data([1:(i-1) (i+1):end], :)], ...
		svm_labels([1:(i-1) (i+1):end]));
  ypred(i) = svmlfwd(latestSVM, svm_data(i,:), svm_labels(i));
end

sum((2 * (ypred > 0) - 1) == svm_labels') / length(svm_labels) 