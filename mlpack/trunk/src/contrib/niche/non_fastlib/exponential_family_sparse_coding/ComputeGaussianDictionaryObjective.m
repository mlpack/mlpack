function f = ComputeGaussianDictionaryObjective(D, S, T)
%function f = ComputeGaussianDictionaryObjective(D, S, T)

n = size(S, 2);

f = -trace(D' * T * S') + 0.5 * norm(D * S, 'fro')^2;

f = f / n; % does this help?
