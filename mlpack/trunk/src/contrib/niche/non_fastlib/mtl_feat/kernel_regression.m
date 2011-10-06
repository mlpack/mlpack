function [a, cost, err, reg] = kernel_regression(K,y,gamma)

n = size(K,1);
a = inv(K+gamma*eye(n)) * y;
cost = gamma * y' * inv(K+gamma*eye(n)) * y;
err = gamma^2 * y' * (inv(K+gamma*eye(n)))^2 * y;
reg = cost - err;
