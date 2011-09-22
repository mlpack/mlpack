function f = ComputePoissonFullObjective(D, S, T, lambda)
%function f = ComputePoissonFullObjective(D, S, T, lambda)

% possibility of numerical overflow
f = -trace(D' * T * S') + sum(sum(exp(D * S))) + lambda * sum(sum(abs(S)))
