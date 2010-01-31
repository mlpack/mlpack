% find_h_E() - find the kernel bandwidth that minimizes entropy
% This is the kernel bandwidth that minimizes the expectation of
% log f_hat_h(X) over the empirical distribution function (the
% empirical distribution function being asymptotically equal to the
% actual density function f(x)

function h_E = find_h_E(X);

N = length(X);

h_E = fmincon(@(h) test_h_E(h, X, N), 1, -1, -1e-4);
