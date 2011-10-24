function f = ComputePoissonSparseCodesObjective(D, s, Dt_t, lambda)
%function f = ComputePoissonSparseCodesObjective(D, s, Dt_t, lambda)
%
% Compute the relevant part of the objective sparse codes in Poisson sparse coding
% Given:
%   Dictionary D
%   Candidate sparse code s
%   Dt_t is D transpose times t, for t the sufficient statistic for the point
%   l1-norm regularization parameter lambda 

f = -s' * Dt_t;

f = f + sum(exp(D * s)); % possibility of numerical overflow

f = f + lambda * sum(abs(s));
