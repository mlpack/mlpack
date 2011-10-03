function [targets, regressors] = ComputeBernoulliTargetsAndRegressors(D, s, t)
%function [targets, regressors] = ComputeBernoulliTargetsAndRegressors(D, s, t)
%
% There is a possibility of numerical under/overflow. This seems to
% unavoidable, so we better hope that D * s is small!


Ds = D * s;

sqrt_exp_Ds = exp(Ds / 2);

targets = sqrt_exp_Ds .* (t - 1) + t - Ds ./ (sqrt_exp_Ds + 1);

regressors = bsxfun(@times, 1 ./ (sqrt_exp_Ds + 1), D);
