function f = ComputeBernoulliFullObjective(D, S, T, lambda)
%function f = ComputeBernoulliFullObjective(D, S, T, lambda)

f = -trace(D' * T * S') + sum(sum(log(1 + exp(D * S)))) + lambda * sum(sum(abs(S)));
