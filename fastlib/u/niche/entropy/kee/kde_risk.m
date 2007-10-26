% kde_risk() - calculate the expected risk of the kernel density
% estimate for the given data and kernel bandwidth
function E_risk = kde_risk(h, X, N);

h

h_star = sqrt(2) *h;
two_h_squared =      2 * (h      ^ 2);
two_h_star_squared = 2 * (h_star ^ 2);
h_sqrt_2_pi =       h      * sqrt(2*pi);
h_star_sqrt_2_pi =  h_star * sqrt(2*pi);

sum = 0;
for i=1:N
  for j=1:N
    sum = sum + exp(-((X(j) - X(i))^2) / two_h_star_squared);
  end
end

int_f_hat_squared = sum / (h_star_sqrt_2_pi * (N^2));

%disp('calculated int_f_hat_squared');


sum = 0;
for i=1:N
  for j=1:N
    sum = sum + exp(-((X(i) - X(j))^2) / two_h_squared);
  end
end

E_f_hat = ...
    (2 / (N - 1)) * gauss_kernel(0, h) - ...
    (2 / (h_sqrt_2_pi * N * (N - 1))) * sum;

%disp('calculated LOOCV estimate of expectation of f_hat');

E_risk = int_f_hat_squared + E_f_hat;
