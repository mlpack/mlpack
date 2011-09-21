function f = ComputeSparseCodesObjective(D, s, Dt_t, lambda)
%function f = ComputeSparseCodesObjective(D, s, t, lambda)
%
% Dictionary D
% Candidate sparse code s
% Dt_t is D transpose times t, where t is the sufficient statistic for a single point
% l1-norm regularization parameter lambda 

f = -s' * Dt_t;

f = f + sum(exp(D * s));

f = f + lambda * sum(abs(s));
