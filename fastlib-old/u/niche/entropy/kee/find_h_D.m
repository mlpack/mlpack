% find_h_D() - find the kernel bandwidth that minimizes risk where
% the loss function is (f(x) f_hat_h(x))^2
function h_D = find_h_D(X);

N = length(X);

h_D = fmincon(@(h) kde_risk(h, X, N), 1, -1, -1e-4);
