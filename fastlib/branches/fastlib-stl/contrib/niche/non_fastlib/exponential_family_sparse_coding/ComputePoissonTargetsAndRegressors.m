function [targets, regressors] = ComputePoissonTargetsAndRegressors(D, s, t)
%function targets = ComputePoissonTargetsAndRegressors(D, s, t)
%
% There is a possibility of numerical under/overflow. This seems to
% unavoidable, so we better hope that D * s is small!


Ds = D * s;

sqrt_exp_Ds = exp(Ds / 2);

targets = t ./ sqrt_exp_Ds + sqrt_exp_Ds .* (Ds - 1);

regressors = bsxfun(@times, sqrt_exp_Ds, D);
