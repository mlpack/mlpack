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


% let's do a retarded random sampling of sigma and C on a grid!


num_sigma_epochs = 100;
num_C_epochs = 100;

loocv_errors = zeros(num_sigma_epochs, num_C_epochs);


sigma = 1e-4;
for sigma_epoch = 1:num_sigma_epochs
  C = 1e-3;
  for C_epoch = 1:num_C_epochs
    
    %  C = 10 * exp(-rand * 10);
    %  sigma = 10 * exp(-rand * 10);
    
    % set option for C (the regularization parameter)
    svm_options = ...
	svmlopt(svm_options, 'KernelParam', sigma, 'C', C);
    
    latestSVM = svml('latestSVM', svm_options);
    
    latestSVM = ...
	svmltrain(latestSVM, 'funknet');
    
    load loocv_error.txt

    C_array(C_epoch) = C;
    C = C * 1.1;
    


    loocv_errors(sigma_epoch, C_epoch) = loocv_error;
    
    %  C = C * 1.01; % geometrically increase C
    %  C_epoch = C_epoch + 1;
  end
  
  sigma_array(sigma_epoch) = sigma;  
  sigma = sigma * 1.1;
end

% for i=1:size(svm_data, 1)
%   latestSVM = ...
%       svmltrain(latestSVM, [svm_data([1:(i-1) (i+1):end], :)], ...
% 		svm_labels([1:(i-1) (i+1):end]));
%   ypred(i) = svmlfwd(latestSVM, svm_data(i,:), svm_labels(i));
% end

% sum((2 * (ypred > 0) - 1) == svm_labels') / length(svm_labels) 



% cluster genes using ICs?

% find a way to do fPCA/fICA using multiple sets of curves
