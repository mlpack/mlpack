function f = ComputeGaussianFullObjective(D, S, T, lambda)
%function f = ComputeGaussianFullObjective(D, S, T, lambda)

f = -trace(D' * T * S') + norm(D * S, 'fro')^2 + lambda * sum(sum(abs(S)));
