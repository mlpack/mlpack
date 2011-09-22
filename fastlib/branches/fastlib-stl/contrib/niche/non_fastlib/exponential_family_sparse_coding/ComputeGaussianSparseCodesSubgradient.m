function subgrad = ComputeGaussianSparseCodesSubgradient(D, s, t, lambda)
%function subgrad = ComputeGaussianSparseCodesSubgradient(D, s, t, lambda)

% choose a subgradient at s
subgrad = -D' * t + D' * D * s;
subgrad = subgrad + lambda * ((s > 0) - (s < 0)); % handle possibly non-differentiable component by using subgradient
