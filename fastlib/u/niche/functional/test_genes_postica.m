clear;
load gene_results;

% discriminant analysis using pc features %
% pc_scores is d x N

g1_indices = find(phases == g1_phase);
nong1_indices = find(phases ~=g1_phase & phases ~= unknown_phase);

used_scores = ic_scores;

used_scores = used_scores([1],:);

svm_data = [used_scores(:,g1_indices) used_scores(:,nong1_indices)]';
svm_labels = [1 * ones(length(g1_indices),1);
	      -1 * ones(length(nong1_indices),1)];



% write data so that we can efficiently call svmltrain many times
svmlwrite('funknet', svm_data, svm_labels);

% set initial svm options %
svm_options = ...
    svmlopt('Kernel', 2, 'KernelParam', 3, 'C', .2, 'ComputeLOO', 1, ...
	    'ExecPath','../../../../matlab/toolboxes/svml');


% let's do a retarded random sampling of sigma and C on a grid!


num_sigma_epochs = 100;
sigma_init = 1e-4;
sigma_grow = 1.1;
%num_order_epochs = 4;
%order_init = 1;
%order_inc = 1;

num_C_epochs = 81;
C_init = 1e-2;
C_grow = 1.1;



loocv_errors = zeros(num_sigma_epochs, num_C_epochs);
%loocv_errors = zeros(num_order_epochs, num_C_epochs);


sigma = sigma_init;
%order = order_init;
for sigma_epoch = 1:num_sigma_epochs
  %for order_epoch = 1:num_order_epochs
  C = C_init;
  for C_epoch = 1:num_C_epochs
    
    %  C = 10 * exp(-rand * 10);
    %  sigma = 10 * exp(-rand * 10);
    
    % set option for C (the regularization parameter)
    svm_options = ...
	svmlopt(svm_options, 'KernelParam', sigma, 'C', C);
    %    svm_options = ...
    %	svmlopt(svm_options, 'KernelParam', order, 'C', C);
    
    
    latestSVM = svml('latestSVM', svm_options);
    
    latestSVM = ...
	svmltrain(latestSVM, 'funknet');
    
    load loocv_error.txt

    C_array(C_epoch) = C;
    C = C * C_grow;
    


    loocv_errors(sigma_epoch, C_epoch) = loocv_error;
    %loocv_errors(order_epoch, C_epoch) = loocv_error;
    
  end
  
  sigma_array(sigma_epoch) = sigma;  
  sigma = sigma * sigma_grow;
  %order_array(order_epoch) = order;  
  %order = order + order_inc;

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



















%{
correct = 0;
same_p = 0;

for i = 1:size(svm_data, 1)
  train_data = svm_data([1:(i-1) (i+1):end]);
  train_labels = svm_labels([1:(i-1) (i+1):end]);
  
  g1_mu = mean(train_data(find(train_labels == 1)));
  g1_sigma = std(train_data(find(train_labels == 1)));
  nong1_mu = mean(train_data(find(train_labels == -1)));
  nong1_sigma = std(train_data(find(train_labels == -1)));
  
  p_g1 = normpdf(svm_data(i), g1_mu, g1_sigma);
  p_nong1 = normpdf(svm_data(i), nong1_mu, nong1_sigma);
  
  if svm_labels(i) == 1
    if p_g1 > p_nong1
      correct = correct + 1;
    end
  else
    if p_nong1 > p_g1
      correct = correct + 1;
    end
  end
  
  if p_g1 == p_nong1
    same_p = same_p + 1;
  end
  
end

correct / size(svm_data,1)
%}
