function f = ComputeSparseCodesObjective(D, s, t, lambda)
%function f = ComputeSparseCodesObjective(D, s, t, lambda)
%
% Dictionary D
% Candidate sparse code s
% Sufficient statistics t for a single point
% l1-norm regularization parameter lambda 

f = -s' * D' * t;

f = f + sum(exp(D * s));

f = f + lambda * sum(abs(s));
