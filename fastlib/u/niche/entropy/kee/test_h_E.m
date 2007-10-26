% test_h_E() - kernel entropy estimation
function kernel_sum = test_h_E(h, X, N);
% select bandwidth h_E that minimizes LOO cross-entropy (expectation of
% log of kernel density estimate over the empirical distribution
% function)

% search for bandwidth that minimizes:
%  \int{(f_hat(x) * log(f_hat(x)))^2 dx}
%  - 2 E[f_hat(x) * log(f_hat(x))]
% evaluate \int{(f_hat(x) * log(f_hat(x)))^2 dx}

% calculate the LOOCV kernel bandwidth
% over log

% first calculate kernel sum

kernel_sum = 0;
  
for i = 1:N
  f_hat_x_i = 0;
  for j = [1:i-1,i+1:N]
    f_hat_x_i = f_hat_x_i + exp(-((X(i) - X(j))^2 / (2 * h^2)));
  end
  f_hat_x_i = f_hat_x_i / (N*h*sqrt(2*pi));
  kernel_sum = kernel_sum + logp_fast(f_hat_x_i);
end

kernel_sum = -kernel_sum / (N - 1);
