function subgrad = ComputeBernoulliSparseCodesSubgradient(D, s, t, lambda)
%function subgrad = ComputeBernoulliSparseCodesSubgradient(D, s, t, lambda)

% choose a subgradient at s
subgrad = -D' * t + D' * (1 ./ (1 + exp(-D * s))); % possibility of numerical overflow
subgrad = subgrad + lambda * ((s > 0) - (s < 0)); % handle possibly non-differentiable component by using subgradient
