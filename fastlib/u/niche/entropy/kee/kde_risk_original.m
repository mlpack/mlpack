% kde_cv() - calculate the expected risk of the kernel density
% estimate for the given data and kernel bandwidth
function [E_risk] = ...
    kde_risk_original(x, h);

N = length(x);

%disp('generated data');

h_star = sqrt(2) *h;
two_h_squared =      2 * (h      ^ 2);
two_h_star_squared = 2 * (h_star ^ 2);
h_sqrt_2_pi =       h      * sqrt(2*pi);
h_star_sqrt_2_pi =  h_star * sqrt(2*pi);

sum = 0;
for i=1:N
  for j=1:N
    sum = sum + exp(-((x(j) - x(i))^2) / two_h_star_squared) / h_star_sqrt_2_pi;
  end
end

int_f_hat_squared = sum / (N^2);

%disp('calculated int_f_hat_squared');


sum = 0;
for i=1:N
  for j=1:N
    sum = sum + exp(-((x(i) - x(j))^2) / two_h_squared) / h_sqrt_2_pi;
  end
end

E_f_hat = ...
    (2 / (N - 1)) * gauss_kernel(0, h) - ...
    (2 / (N * (N - 1))) * sum;

%disp('calculated LOOCV estimate of expectation of f_hat');

E_risk = int_f_hat_squared + E_f_hat;
