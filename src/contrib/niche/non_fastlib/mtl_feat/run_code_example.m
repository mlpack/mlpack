function [testerrs,theW,theD] = run_code_example(gammas,trainx,trainy,testx,testy,task_indexes,task_indexes_test,Dini,iterations, method_str, epsilon_init, fname)
%function [testerrs,theW,theD] = run_code_example(gammas,trainx,trainy,testx,testy,task_indexes,task_indexes_test,Dini,iterations, method_str, epsilon_init, fname)

% Example script of running the train_alternating_epsilon() code
% Uses regularizer trace( W' D^{-1} W ) and square loss (see file
% kernel_regression.m)

% INPUTS:
% gammas: a vector of the gammas to select from using cross-validation
% task_indexes: starting indexes for each task in data 
% cv_size: number of cross validations performed 
% Dini: the initial matrix D 
% epsilon_init : perturbation epsilon (use 0 for no perturbation)
% method_str : see below
% See train_alternating.m for the rest

% OUTPUTS:
% bestcv: the best cross-validation performance among the gammas tried
% bestgamma: the best gamma found using cross-validation
% theW: a matrix for which each column is a w_t
% theD: matrix D estimated

feat = 1; independent = 2; diagonal = 3;

if (strcmp(method_str,'feat'))
    method = feat;
elseif (strcmp(method_str,'ind'))
    method = independent;
elseif (strcmp(method_str,'diag'))
    method = diagonal;
else
    error('Wrong method');
end
% (see file train_alternating.m)

% Define method for computing f(D)
function v = vec_inv(d)
    v = zeros(length(d),1);
    ind = find(d > eps);
    v(ind) = 1 ./ d(ind);
end

[theW,theD,costs,mineps] = train_alternating_epsilon(trainx, trainy, task_indexes, gammas, Dini, iterations, ...
    method, 'kernel_regression', @vec_inv, @(b)(b/sum(b)), epsilon_init);

testerrs = mean(test_error_unbalanced(theW,testx,testy,task_indexes_test));

save(sprintf('results_%s_%s_lin.mat',fname,method_str),'gammas','Dini','method_str',...
    'testerrs','theW','theD','costs','mineps');

end
