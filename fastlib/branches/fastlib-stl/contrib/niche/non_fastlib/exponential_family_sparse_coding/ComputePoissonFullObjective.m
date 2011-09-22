function f = ComputePoissonFullObjective(D, S, T, lambda)
%function f = ComputePoissonFullObjective(D, S, T, lambda)

f = -trace(D' * T * S') + sum(sum(exp(D * S))) + lambda * sum(sum(abs(S)))
