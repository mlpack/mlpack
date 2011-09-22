function subgrad = ComputePoissonSparseCodesSubgradient(D, s, t, lambda)
%function subgrad = ComputePoissonSparseCodesSubgradient(D, s, t)

% t = T(:,i)

% choose a subgradient at s
subgrad = -D' * t + D' * exp(D * s);
subgrad = subgrad + lambda * ((s > 0) - (s < 0)); % handle possibly non-differentiable component by using subgradient
