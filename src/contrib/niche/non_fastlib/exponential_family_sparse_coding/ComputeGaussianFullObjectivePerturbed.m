function f = ComputeGaussianFullObjectivePerturbed(D, Q, S, T, lambda_S, lambda_Q)
%function f = ComputeGaussianFullObjectivePerturbed(D, Q, S, T, lambda_S, lambda_Q)

E = D + Q;

f = -trace(E' * T * S') + 0.5 * norm(E * S, 'fro')^2 ...
    + lambda_S * sum(sum(abs(S))) ...
    + lambda_Q * sum(sum(abs(Q)));
