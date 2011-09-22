function f = ComputeFullObjective(D, S, T, lambda)
%function f = ComputeFullObjective(D, S, T, lambda)

n = size(S, 2);

f = -trace(D' * T * S') + sum(sum(exp(D * S))) + lambda * sum(sum(abs(S)))
